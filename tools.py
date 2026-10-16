"""Deterministic functions behind the agent's tools.

The agent and the eval share these, so every eval answer can be computed without an LLM.
Scene bands are cached in data/scenes/, so each (reach, month) is downloaded once.

Usage: uv run tools.py   (fills the cache for every reach, July and September 2023-2026)
"""
import calendar
from functools import cache
from pathlib import Path

import numpy as np

import s2
from dataset import REACHES

CACHE = Path("data/scenes")
PIXEL_KM2 = 1e-4  # one 10 m x 10 m pixel
METHODS = ("unet", "baseline")


def reaches():
    """Reaches ordered upstream to downstream (the river flows north, so by latitude)."""
    order = sorted(REACHES, key=lambda r: REACHES[r][1])
    return [{"reach": r, "lon": REACHES[r][0], "lat": REACHES[r][1], "order_from_upstream": i + 1} for i, r in enumerate(order)]


def bbox(reach):
    lon, lat = REACHES[reach]
    return [lon - 0.09, lat - 0.08, lon + 0.09, lat + 0.08]


def scene(reach, year, month):
    """Date, cloud cover and bands of the month's least cloudy scene (<10% cloud)."""
    if reach not in REACHES:
        raise ValueError(f"unknown reach {reach!r}; known reaches: {', '.join(REACHES)}")
    path = CACHE / f"{reach}_{year}-{month:02d}.npz"
    if not path.exists():
        last = calendar.monthrange(year, month)[1]
        items = s2.search(bbox(reach), f"{year}-{month:02d}-01/{year}-{month:02d}-{last}")
        if not items:
            raise ValueError(f"no Sentinel-2 scene under 10% cloud for {reach} in {year}-{month:02d}")
        item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
        b = s2.load_day(items, item, bbox(reach))
        CACHE.mkdir(parents=True, exist_ok=True)
        bands = {k: v.astype("float16") for k, v in b.items()}
        tmp = path.with_suffix(".tmp.npz")  # write then rename, so a concurrent reader never sees half a file
        np.savez_compressed(tmp, date=str(item.datetime.date()), cloud=item.properties["eo:cloud_cover"], **bands)
        tmp.replace(path)
    d = np.load(path)
    return str(d["date"]), float(d["cloud"]), {k: d[k].astype("float32") for k in s2.BANDS}


@cache
def _model():
    import predict  # torch and the checkpoint load only when the U-Net is asked for

    if not Path("unet.pt").exists():
        raise ValueError("U-Net checkpoint unet.pt not trained yet; use method='baseline'")
    return predict.predict


@cache
def measure(reach, year, month, method="unet"):
    """Exposed sand and water area for one reach and month."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    date, cloud, b = scene(reach, year, month)
    if method == "baseline":
        water, sand = s2.baseline(b)
    else:
        pred = _model()(b)
        water, sand = pred == 1, pred == 2
    return {
        "reach": reach,
        "scene_date": date,
        "cloud_cover_pct": round(cloud, 2),
        "method": method,
        "sand_km2": round(float(sand.sum()) * PIXEL_KM2, 3),
        "water_km2": round(float(water.sum()) * PIXEL_KM2, 3),
        "valid_fraction": round(float((~np.isnan(b["B04"])).mean()), 3),
    }


if __name__ == "__main__":
    assert [r["reach"] for r in reaches()][:2] == ["barra-do-garcas", "aruana"]
    for r in REACHES:
        for y in (2023, 2024, 2025, 2026):
            for m in (7, 9):
                try:
                    print(measure(r, y, m, "baseline"), flush=True)
                except ValueError as e:
                    print(e, flush=True)

"""Build the weak-label dataset: 256x256 tiles along the Araguaia with baseline masks.

Labels: 0 = other, 1 = water, 2 = sand, 255 = no data (ignored in training).
Writes one data/<reach>_<date>.npz per scene. Already written scenes are skipped.

Usage: uv run dataset.py [reach ...]   (default: all reaches)
"""
import sys
from pathlib import Path

import numpy as np

import s2

# Reaches along the Araguaia, (lon, lat) of the center. Each box is ~20 x 18 km.
REACHES = {
    "barra-do-garcas": (-52.26, -15.89),
    "aruana": (-51.07, -14.92),
    "cocalinho": (-51.00, -14.39),
    "luiz-alves": (-50.58, -13.22),
    "sao-felix": (-50.67, -11.62),
    "santa-terezinha": (-50.51, -10.47),
    "conceicao": (-49.26, -8.26),
    "xambioa": (-48.53, -6.41),
}
# Two dry-season windows per year: early (July) and late (September).
WINDOWS = [f"{y}-{m}" for y in (2023, 2024, 2025, 2026) for m in ("07-01/07-31", "09-01/09-30")]
TILE = 256
OUT = Path("data")


def tiles(b, water, sand):
    x = np.stack([b[k] for k in s2.BANDS])
    y = np.where(sand, 2, np.where(water, 1, 0)).astype("uint8")
    y[np.isnan(x).any(0)] = 255
    x = np.nan_to_num(x).astype("float16")
    xs, ys = [], []
    for r in range(0, y.shape[0] - TILE + 1, TILE):
        for c in range(0, y.shape[1] - TILE + 1, TILE):
            t = y[r : r + TILE, c : c + TILE]
            # Keep tiles on the river with mostly valid pixels.
            if (t == 255).mean() < 0.1 and (t == 1).mean() > 0.02:
                xs.append(x[:, r : r + TILE, c : c + TILE])
                ys.append(t)
    return np.array(xs), np.array(ys)


def main():
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or REACHES:
        lon, lat = REACHES[name]
        bbox = [lon - 0.09, lat - 0.08, lon + 0.09, lat + 0.08]
        for w in WINDOWS:
            year, span = w.split("-", 1)
            start, end = span.split("/")
            items = s2.search(bbox, f"{year}-{start}/{year}-{end}", max_cloud=5)
            if not items:
                print(f"{name} {year}-{start}: no clear scene")
                continue
            item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
            path = OUT / f"{name}_{item.datetime.date()}.npz"
            if path.exists():
                continue
            b = s2.load_day(items, item, bbox)
            water, sand = s2.baseline(b)
            x, y = tiles(b, water, sand)
            if len(x) == 0:
                print(f"{path.name}: no river tiles")
                continue
            np.savez_compressed(path, x=x, y=y)
            print(f"{path.name}: {len(x)} tiles, sand {(y == 2).mean():.1%}", flush=True)


if __name__ == "__main__":
    main()

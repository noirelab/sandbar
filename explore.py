"""Fetch the least cloudy Sentinel-2 L2A day over an AOI and plot the spectral baseline.

Usage: uv run explore.py [YYYY-MM-DD/YYYY-MM-DD] [minlon,minlat,maxlon,maxlat]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import s2

# Default AOI: Araguaia River at Aruanã, GO.
DATES = sys.argv[1] if len(sys.argv) > 1 else "2026-07-01/2026-09-30"
BBOX = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [-51.16, -15.00, -50.98, -14.84]

items = s2.search(BBOX, DATES)
print(f"{len(items)} scenes with <10% cloud")
item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
print("using", item.id, item.datetime.date(), f"{item.properties['eo:cloud_cover']:.2f}% cloud")

b = s2.load_day(items, item, BBOX)
mndwi, _ = s2.indices(b)
water, sand = s2.baseline(b)

rgb = np.nan_to_num(np.clip(np.dstack([b["B04"], b["B03"], b["B02"]]) / 0.3, 0, 1))
overlay = rgb.copy()
overlay[water] = 0.6 * overlay[water] + 0.4 * np.array([0, 0.3, 1])
overlay[sand] = np.array([1, 0.85, 0])

fig, ax = plt.subplots(1, 3, figsize=(18, 7))
for a, img, title in zip(ax, [rgb, mndwi, overlay], ["RGB", "MNDWI", "baseline: water (blue), sand (yellow)"]):
    a.imshow(img, cmap="RdBu" if img.ndim == 2 else None)
    a.set_title(title)
    a.axis("off")
plt.tight_layout()
out = f"baseline_{item.datetime.date()}.png"
plt.savefig(out, dpi=110)
print(f"saved {out}  water {water.mean():.1%}  sand {sand.mean():.1%}")

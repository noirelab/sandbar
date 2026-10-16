"""Phase 1: fetch one Sentinel-2 L2A scene over an AOI and build a spectral sand/water baseline.

Usage: uv run explore.py [YYYY-MM-DD/YYYY-MM-DD] [minlon,minlat,maxlon,maxlat]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from scipy import ndimage
from odc.stac import load

# Default AOI: lower Piquiri River, PR, a few km above its mouth on the Paraná.
DATES = sys.argv[1] if len(sys.argv) > 1 else "2026-06-01/2026-09-30"
BBOX = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [-54.12, -24.15, -53.88, -23.97]

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
items = list(
    catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATES,
        query={"eo:cloud_cover": {"lt": 10}},
    ).items()
)
print(f"{len(items)} scenes with <10% cloud")
item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
print("using", item.id, item.datetime.date(), f"{item.properties['eo:cloud_cover']:.2f}% cloud")

ds = load([item], bands=["B02", "B03", "B04", "B08", "B11"], bbox=BBOX, resolution=10).isel(time=0)
# Processing baseline 04.00+ (scenes from 2022 on) stores reflectance with a +1000 offset.
offset = 1000 if item.properties["s2:processing_baseline"] >= "04.00" else 0
b = {k: (ds[k].values.astype("float32") - offset) / 10000 for k in ds.data_vars}

eps = 1e-6
mndwi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"] + eps)
ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)

# Baseline rules: water = MNDWI > 0; bare sand = not water, little vegetation, bright in visible.
# Sandbars touch the river, so sand is limited to a band around the main water body
# (drops towns, roads and bare fields that share the same spectrum).
water = mndwi > 0
labels, _ = ndimage.label(water)
river = labels == np.bincount(labels.ravel())[1:].argmax() + 1
near_river = ndimage.binary_dilation(river, iterations=15)  # 15 px = 150 m
sand = ~water & near_river & (ndvi < 0.15) & (b["B04"] > 0.12)

rgb = np.clip(np.dstack([b["B04"], b["B03"], b["B02"]]) / 0.3, 0, 1)
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

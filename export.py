"""Precompute everything the web demo shows, so it runs without a GPU or the scene cache.

Writes demo/measurements.json (U-Net and rule results for every cached scene) and one
overlay image per scene: RGB with U-Net sand in yellow and water tinted blue.

Usage: uv run export.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

import predict
import s2
import tools

OUT = Path("demo")
STEP = 3  # 30 m per image pixel keeps each overlay around 100 KB


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(tools.CACHE.glob("*.npz")):
        reach, ym = path.stem.split("_")
        year, month = map(int, ym.split("-"))
        for method in tools.METHODS:
            rows.append(tools.measure(reach, year, month, method) | {"year": year, "month": month})
        _, _, b = tools.scene(reach, year, month)
        pred = predict.predict(b)[::STEP, ::STEP]
        rgb = np.nan_to_num(np.clip(np.dstack([b["B04"], b["B03"], b["B02"]])[::STEP, ::STEP] / 0.3, 0, 1))
        img = predict.overlay(rgb, pred == 1, pred == 2)
        Image.fromarray((img * 255).astype("uint8")).save(OUT / "img" / f"{path.stem}.jpg", quality=80)
        print(path.stem, flush=True)
    (OUT / "measurements.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} measurements")


if __name__ == "__main__":
    main()

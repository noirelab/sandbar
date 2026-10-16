"""Run the trained U-Net on a full scene and plot it next to the spectral baseline.

Usage: uv run predict.py [YYYY-MM-DD/YYYY-MM-DD] [minlon,minlat,maxlon,maxlat]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

import s2

# Default: Aruanã, the reach held out of training.
DATES = "2026-07-01/2026-09-30"
BBOX = [-51.16, -15.00, -50.98, -14.84]


def predict(b, ckpt="unet.pt"):
    state = torch.load(ckpt)
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=5, classes=3).cuda().eval()
    model.load_state_dict(state["model"])
    x = torch.from_numpy(np.nan_to_num(np.stack([b[k] for k in s2.BANDS]))).float()
    x = (x - state["mean"][:, None, None]) / state["std"][:, None, None]
    h, w = x.shape[1:]
    x = F.pad(x[None], (0, -w % 32, 0, -h % 32), mode="reflect").cuda()  # U-Net needs multiples of 32
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x).argmax(1)[0, :h, :w].cpu().numpy()


def overlay(rgb, water, sand):
    out = rgb.copy()
    out[water] = 0.6 * out[water] + 0.4 * np.array([0, 0.3, 1])
    out[sand] = np.array([1, 0.85, 0])
    return out


def main():
    dates = sys.argv[1] if len(sys.argv) > 1 else DATES
    bbox = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else BBOX
    items = s2.search(bbox, dates)
    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    b = s2.load_day(items, item, bbox)
    water, sand = s2.baseline(b)
    pred = predict(b)

    rgb = np.nan_to_num(np.clip(np.dstack([b["B04"], b["B03"], b["B02"]]) / 0.3, 0, 1))
    fig, ax = plt.subplots(1, 3, figsize=(18, 7))
    panels = [rgb, overlay(rgb, water, sand), overlay(rgb, pred == 1, pred == 2)]
    for a, img, title in zip(ax, panels, ["RGB", "spectral baseline", "U-Net"]):
        a.imshow(img)
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    out = f"predict_{item.datetime.date()}.png"
    plt.savefig(out, dpi=110)
    agree = ((pred == 2) & sand).sum() / max(((pred == 2) | sand).sum(), 1)
    print(f"saved {out}  sand IoU model vs baseline {agree:.3f}")


if __name__ == "__main__":
    main()

"""Train a U-Net on the weak-label tiles in data/.

Reaches in HOLDOUT never enter training: "sao-felix" is validation, "aruana" is kept
for the hand-labeled test set. Validation IoU is measured against weak labels, so it
shows agreement with the baseline rule, not accuracy.

Usage: uv run train.py [epochs]
"""
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

VAL, TEST = "sao-felix", "aruana"
CLASSES = ["other", "water", "sand"]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
dev = "cuda"


def load(pred):
    files = [f for f in sorted(Path("data").glob("*.npz")) if pred(f.name.split("_")[0])]
    xs, ys = zip(*[(d["x"], d["y"]) for d in map(np.load, files)])
    return torch.from_numpy(np.concatenate(xs)).float(), torch.from_numpy(np.concatenate(ys)).long()


def normalize(x, mean, std):
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def augment(x, y):
    k = int(torch.randint(4, ()))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if torch.rand(()) < 0.5:
        x, y = x.flip(3), y.flip(2)
    return x, y


def iou(pred, y):
    valid = y != 255
    return [
        ((pred == c) & (y == c) & valid).sum().item() / max((((pred == c) | (y == c)) & valid).sum().item(), 1)
        for c in range(len(CLASSES))
    ]


def main():
    xtr, ytr = load(lambda r: r not in (VAL, TEST))
    xva, yva = load(lambda r: r == VAL)
    print(f"train {len(xtr)} tiles, val {len(xva)} tiles")
    mean, std = xtr.mean((0, 2, 3)), xtr.std((0, 2, 3))
    xtr, xva = normalize(xtr, mean, std), normalize(xva, mean, std)

    # Sand is ~1-2% of pixels, so weight classes by inverse frequency.
    freq = torch.bincount(ytr[ytr != 255], minlength=3).float()
    weight = (freq.sum() / freq).sqrt().to(dev)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=5, classes=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * (len(xtr) // 16 + 1))
    best = -1
    for epoch in range(EPOCHS):
        model.train()
        for i in torch.randperm(len(xtr)).split(16):
            x, y = augment(xtr[i].to(dev), ytr[i].to(dev))
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y, weight=weight, ignore_index=255)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            pred = torch.cat([model(xva[i].to(dev)).argmax(1).cpu() for i in torch.arange(len(xva)).split(32)])
        scores = iou(pred, yva)
        print(f"epoch {epoch + 1:2d}  loss {loss.item():.3f}  " + "  ".join(f"{c} {s:.3f}" for c, s in zip(CLASSES, scores)), flush=True)
        if scores[2] > best:
            best = scores[2]
            torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "unet.pt")
    print(f"best val sand IoU vs weak labels: {best:.3f}")


if __name__ == "__main__":
    main()

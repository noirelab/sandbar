"""Sentinel-2 access and the rule-based water/sand labels shared by every script."""
import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load
from scipy import ndimage

BANDS = ["B02", "B03", "B04", "B08", "B11"]  # blue, green, red, NIR, SWIR1

_catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def search(bbox, dates, max_cloud=10):
    return list(
        _catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=dates,
            query={"eo:cloud_cover": {"lt": max_cloud}},
        ).items()
    )


def load_day(items, item, bbox):
    """Reflectance per band for the day of `item`, NaN where there is no data."""
    # An AOI can straddle tile edges, so mosaic every tile from the chosen day.
    same_day = [i for i in items if i.datetime.date() == item.datetime.date()]
    ds = load(same_day, bands=BANDS, bbox=bbox, resolution=10, groupby="solar_day").isel(time=0)
    # Processing baseline 04.00+ (scenes from 2022 on) stores reflectance with a +1000 offset.
    offset = 1000 if item.properties["s2:processing_baseline"] >= "04.00" else 0
    b = {}
    for k in BANDS:
        raw = ds[k].values.astype("float32")
        b[k] = np.where(raw == 0, np.nan, (raw - offset) / 10000)
    return b


def indices(b):
    eps = 1e-6
    mndwi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"] + eps)
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    return mndwi, ndvi


def baseline(b):
    """Rule-based masks: water = MNDWI > 0; sand = not water, little vegetation,
    bright in red, and within 150 m of the main river body (drops towns, roads and
    bare fields that share the spectrum of dry sand)."""
    mndwi, ndvi = indices(b)
    water = mndwi > 0
    labels, n = ndimage.label(water)
    if n == 0:
        return water, np.zeros_like(water)
    river = labels == np.bincount(labels.ravel())[1:].argmax() + 1
    near_river = ndimage.binary_dilation(river, iterations=15)  # 15 px = 150 m
    sand = ~water & near_river & (ndvi < 0.15) & (b["B04"] > 0.12)
    return water, sand

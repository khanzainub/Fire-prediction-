"""
raster_utils.py
----------------
Helper functions:
  1. GeoTIFF static layers ko ek common grid pe align/resample karna
     (rasterio ke through) - kyunki curing (30m), LAI (500m), slope (30m),
     fuel (10m) sab alag native resolution ke hain.
  2. Open-Meteo block-wise weather ko us grid pe per-cell broadcast karna.
  3. CA state snapshots ko PNG frames me render karna (web animation ke liye).
"""

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image

from ca_engine import UNBURNED, BURNING, BURNT


def load_and_align(reference_path: Path, target_path: Path) -> np.ndarray:
    """
    target_path ki raster ko reference_path ke grid (shape, transform, crs) pe
    resample karke numpy array return karta hai. reference = usually curing_index
    (sabse important variable, baaki isi ke grid pe align hote hain).
    """
    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    with rasterio.open(target_path) as src:
        dst = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    return dst


def load_reference_grid(reference_path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(reference_path) as ref:
        arr = ref.read(1).astype(np.float32)
        meta = ref.meta.copy()
    return arr, meta


def broadcast_weather_to_grid(weather_blocks: List[Dict], grid_shape: Tuple[int, int],
                               grid_transform, var_path: str) -> np.ndarray:
    """
    weather_blocks: output of weather_fetch.fetch_weather_for_grid(...) - har block
    me 'bounds' (west, south, east, north) aur 'weather' data hai.
    var_path: which hourly variable to extract for "current" timestep, e.g.
              'hourly.wind_speed_10m' (index[-1] = latest available hour taken by caller)

    Simplistic nearest-block broadcast: har grid cell ko uske containing block
    ki value assign karo. Fine hai kyunki NWP resolution khud coarse hai.
    """
    grid = np.zeros(grid_shape, dtype=np.float32)
    rows, cols = grid_shape
    # cell center coords nikaalo transform se
    xs = np.array([grid_transform * (c + 0.5, 0) for c in range(cols)])[:, 0]
    ys = np.array([grid_transform * (0, r + 0.5) for r in range(rows)])[:, 1]

    for block in weather_blocks:
        w, s, e, n = block["bounds"]
        row_mask = (ys >= s) & (ys < n)
        col_mask = (xs >= w) & (xs < e)
        if not row_mask.any() or not col_mask.any():
            continue
        value = _extract_var(block["weather"], var_path)
        if value is None:
            continue
        rr = np.where(row_mask)[0]
        cc = np.where(col_mask)[0]
        grid[np.ix_(rr, cc)] = value
    return grid


def _extract_var(weather_json: dict, var_path: str):
    """weather_json['hourly'][var][-1] jaisa dotted path resolve karta hai."""
    parts = var_path.split(".")
    node = weather_json
    for p in parts[:-1]:
        node = node.get(p, {})
    arr = node.get(parts[-1]) if isinstance(node, dict) else None
    if not arr:
        return None
    return arr[-1]  # latest hour - caller time-index control chahe to isko extend karo


STATE_COLORS = {
    UNBURNED: (34, 87, 46),     # forest green
    BURNING: (217, 95, 50),     # ember orange
    BURNT: (40, 38, 38),        # charred black-brown
}


def render_state_png(state: np.ndarray, out_path: Path, scale: int = 2) -> Path:
    """CA grid state ko ek color-coded PNG frame me render karo (animation ke liye)."""
    h, w = state.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in STATE_COLORS.items():
        rgb[state == val] = color
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    img.save(out_path)
    return out_path


def render_animation_frames(history, out_dir: Path, prefix: str = "frame") -> List[str]:
    """CAModel.run() ke history (list of GridState) ko PNG frames ki series bana do."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, gs in enumerate(history):
        p = out_dir / f"{prefix}_{i:04d}.png"
        render_state_png(gs.state, p)
        paths.append(p.name)
    return paths

"""
main.py
--------
FastAPI app - web tool ka backend. Frontend (static/index.html) yahi
endpoints call karta hai.

Run locally:
    uvicorn main:app --reload --port 8000

Environment variables set karna mat bhoolna (README.md dekho):
    FIRMS_MAP_KEY, GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY_FILE
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

import static_layers
import weather_fetch
import ignition_fetch
import raster_utils
from ca_engine import CAModel, CAParams
from config import OUTPUTS_DIR, STATIC_LAYERS_DIR, DEFAULT_CELL_SIZE_M, WEATHER_BLOCK_SIZE

app = FastAPI(title="Wildfire CA Prediction Tool")


# ---------- request schemas ----------

class BBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

    def as_tuple(self):
        return (self.west, self.south, self.east, self.north)


class RunRequest(BaseModel):
    bbox: BBox
    mode: str                      # 'hindcast' | 'nowcast' | 'forecast'
    start_date: Optional[str] = None   # 'YYYY-MM-DD' - hindcast ke liye fire ka known start
    end_date: Optional[str] = None     # hindcast validation end date
    fire_source: str = "viirs_snpp"    # FIRMS source key
    n_steps: int = 24                  # kitne CA timesteps chalane hain
    force_refresh_static: bool = False


# ---------- static layer endpoints ----------

@app.post("/static-layers/prepare")
def prepare_static_layers(bbox: BBox, force_refresh: bool = False):
    """
    Curing index, LAI, slope/aspect, fuel-landcover - sab GEE se fetch/cache karo.
    Yeh call sabse pehle karo naye AOI ke liye (cache hone ke baad baar-baar
    fast rahega, TTL ke andar).
    """
    try:
        paths = static_layers.get_all_static_layers(bbox.as_tuple(), force_refresh=force_refresh)
        return {"status": "ok", "layers": paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GEE fetch failed: {e}")


@app.get("/static-layers/download/{layer_name}")
def download_static_layer(layer_name: str, west: float, south: float, east: float, north: float):
    """Kisi bhi cached static-layer GeoTIFF ko download karo."""
    bbox = (west, south, east, north)
    getters = {
        "curing_index": static_layers.get_curing_index,
        "lai": static_layers.get_lai,
        "slope_aspect": static_layers.get_slope_aspect,
        "fuel_landcover": static_layers.get_fuel_landcover,
    }
    if layer_name not in getters:
        raise HTTPException(status_code=404, detail="unknown layer name")
    path = getters[layer_name](bbox)
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="layer not cached yet - call /static-layers/prepare first")
    return FileResponse(path, filename=f"{layer_name}.tif")


# ---------- ignition endpoint ----------

@app.get("/ignition/latest")
def latest_ignition(west: float, south: float, east: float, north: float,
                     source: str = "viirs_snpp", day_range: int = 2):
    """FIRMS se AOI ke andar sabse recent/earliest fire detection nikaalo."""
    try:
        result = ignition_fetch.find_ignition_seed((west, south, east, north), source, day_range)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- main simulation endpoint ----------

@app.post("/run")
def run_simulation(req: RunRequest):
    """
    Ek CA simulation chalao: mode ke hisaab se ignition + weather source decide
    hoga, static layers cache se load honge, aur output PNG-frame animation +
    final-state GeoTIFF banega.
    """
    bbox = req.bbox.as_tuple()

    # 1. Static layers (cache se, ya fetch karo agar missing)
    try:
        layer_paths = static_layers.get_all_static_layers(bbox, force_refresh=req.force_refresh_static)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"static layer error: {e}")

    curing_path = Path(layer_paths["curing_index"])
    if not curing_path.exists():
        raise HTTPException(status_code=500, detail="curing_index raster missing after prepare - GEE export fail hua")

    curing_arr, meta = raster_utils.load_reference_grid(curing_path)
    lai_arr = raster_utils.load_and_align(curing_path, layer_paths["lai"])
    slope_aspect_arr = raster_utils.load_and_align(curing_path, layer_paths["slope_aspect"])
    slope_arr = slope_aspect_arr  # single band read - agar 2-band export hai to yahan band-specific load use karo
    aspect_arr = slope_aspect_arr

    # 2. Ignition seeding
    day_range = 10 if req.mode == "hindcast" else 2
    try:
        ignition = ignition_fetch.find_ignition_seed(bbox, req.fire_source, day_range, req.end_date)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ignition:
        raise HTTPException(status_code=404, detail="AOI/time-window me koi FIRMS detection nahi mila")

    seed_lat, seed_lon = ignition["seed"]["lat"], ignition["seed"]["lon"]
    transform = meta["transform"]
    seed_col, seed_row = ~transform * (seed_lon, seed_lat)
    ignition_cells = [(int(seed_row), int(seed_col))]

    # 3. Weather fetch, mode ke hisaab se
    cell_size_deg = abs(transform.a)  # approx - proper deg-per-pixel conversion AOI ke CRS pe depend karta hai
    if req.mode in ("hindcast", "nowcast"):
        start_date = req.start_date or ignition["seed"]["acq_datetime"][:10]
        end_date = req.end_date or datetime.utcnow().strftime("%Y-%m-%d")
        weather_blocks = weather_fetch.fetch_weather_for_grid(
            bbox, req.mode, cell_size_deg, WEATHER_BLOCK_SIZE, start_date=start_date, end_date=end_date
        )
    elif req.mode == "forecast":
        forecast_days = max(1, req.n_steps // 24 + 1)
        weather_blocks = weather_fetch.fetch_weather_for_grid(
            bbox, "forecast", cell_size_deg, WEATHER_BLOCK_SIZE, forecast_days=forecast_days
        )
    else:
        raise HTTPException(status_code=400, detail="mode must be hindcast|nowcast|forecast")

    wind_speed_grid = raster_utils.broadcast_weather_to_grid(
        weather_blocks, curing_arr.shape, transform, "weather.hourly.wind_speed_10m"
    )
    wind_dir_grid = raster_utils.broadcast_weather_to_grid(
        weather_blocks, curing_arr.shape, transform, "weather.hourly.wind_direction_10m"
    )
    # NOTE: yeh sirf latest available hour use karta hai har block ke liye (see
    # raster_utils._extract_var). Multi-step run ke liye per-timestep wind_series
    # banao (hourly index loop karke) - abhi simplistic single-frame demo hai,
    # extend karne ka sabse pehla TODO yahi hai.
    wind_series = [{"speed": wind_speed_grid, "direction": wind_dir_grid}] * req.n_steps

    # 4. CA run
    model = CAModel(curing_arr, lai_arr, slope_arr, aspect_arr, params=CAParams())
    gs0 = model.init_state(ignition_cells)
    history = model.run(gs0, req.n_steps, wind_series)

    # 5. Output: animation frames + final raster
    run_id = f"{req.mode}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    run_dir = OUTPUTS_DIR / run_id
    frame_names = raster_utils.render_animation_frames(history, run_dir)

    final_state_path = run_dir / "final_state.npy"
    np.save(final_state_path, history[-1].state)

    return {
        "run_id": run_id,
        "ignition_seed": ignition["seed"],
        "n_frames": len(frame_names),
        "frames_base_url": f"/outputs/{run_id}/",
        "frames": frame_names,
        "final_state_download": f"/outputs-download/{run_id}/final_state.npy",
    }


# ---------- static files / outputs serving ----------

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/outputs-download/{run_id}/{filename}")
def download_output(run_id: str, filename: str):
    path = OUTPUTS_DIR / run_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=filename)


@app.get("/runs")
def list_runs():
    """Sab past simulation runs list karo (download/re-view ke liye)."""
    runs = []
    for d in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            frames = sorted(p.name for p in d.glob("frame_*.png"))
            runs.append({"run_id": d.name, "n_frames": len(frames)})
    return {"runs": runs}


# ---------- frontend ----------

app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static-frontend")

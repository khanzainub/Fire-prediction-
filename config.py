"""
config.py
---------
Sab settings ek jagah. API keys environment variables se aati hain
(kabhi bhi code me hardcode mat karna, especially FIRMS MAP_KEY aur
GEE service account path).

Required env vars (deployment ke waqt set karo):
    FIRMS_MAP_KEY        -> https://firms.modis.earthdata.nasa.gov/api/map_key/
                            (free, sirf ek email se milti hai)
    GEE_SERVICE_ACCOUNT  -> GEE service account email (agar server-side auth use kar rahe ho)
    GEE_PRIVATE_KEY_FILE -> uss service account ki .json key file ka path

Open-Meteo ke liye koi key nahi chahiye - fully public/free API hai.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
STATIC_LAYERS_DIR = CACHE_DIR / "static_layers"   # curing, LAI, slope, fuel (rarely re-run)
WEATHER_CACHE_DIR = CACHE_DIR / "weather"          # per grid-cell weather json, TTL-based
IGNITION_CACHE_DIR = CACHE_DIR / "ignitions"       # FIRMS point pulls
OUTPUTS_DIR = CACHE_DIR / "outputs"                # CA simulation results (rasters + frames)

for d in (CACHE_DIR, STATIC_LAYERS_DIR, WEATHER_CACHE_DIR, IGNITION_CACHE_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Cache TTLs (seconds) ---
# Static layers (curing index, LAI, slope, fuel type) badalte bahut slow hain,
# isliye lambi TTL - user chahe to /refresh-static-layers endpoint se force-refresh kar sakta hai.
STATIC_LAYER_TTL = 60 * 60 * 24 * 30      # 30 din
WEATHER_ARCHIVE_TTL = 60 * 60 * 24 * 365  # historical weather kabhi nahi badalta - practically forever
WEATHER_FORECAST_TTL = 60 * 30            # forecast har 30 min me refresh
IGNITION_TTL = 60 * 15                    # FIRMS naye pass har ~15-20 min (VIIRS) - jaldi refresh

# --- External APIs ---
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
FIRMS_BASE_URL = "https://firms.modis.earthdata.nasa.gov/api"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

GEE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_PRIVATE_KEY_FILE = os.environ.get("GEE_PRIVATE_KEY_FILE", "")
GEE_SERVICE_ACCOUNT_KEY_JSON = os.environ.get("GEE_SERVICE_ACCOUNT_KEY_JSON", "")

# --- CA grid defaults ---
DEFAULT_CELL_SIZE_M = 30       # Sentinel-2/Landsat native resolution ke saath match
WEATHER_BLOCK_SIZE = 5         # har 5x5 cell block ek Open-Meteo query share karega

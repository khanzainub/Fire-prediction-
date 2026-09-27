"""
weather_fetch.py
-----------------
Open-Meteo se live/dynamic conditioning variables: wind speed, wind
direction, air temperature, precipitation, humidity.

Do endpoints:
  - Archive API   -> hindcast aur nowcast ("ab tak kya hua") ke liye
  - Forecast API  -> forecast mode ke liye ("aage kya hoga")

Koi API key nahi chahiye - Open-Meteo fully public hai.

Caching strategy: grid ko 5x5-cell blocks me todte hain (config.WEATHER_BLOCK_SIZE),
har block ka centroid nikal ke ek hi API call karte hain (poore block ke liye same
weather maan lete hain - resolution se zyada fine-grain karne ka koi fayda nahi
kyunki NWP models khud bhi ~9-25km resolution ke hote hain).
"""

from typing import List, Dict, Tuple
import requests

from config import (
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    WEATHER_ARCHIVE_TTL,
    WEATHER_FORECAST_TTL,
    WEATHER_CACHE_DIR,
)
from cache_manager import DiskCache, make_key

_cache = DiskCache(WEATHER_CACHE_DIR)

HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"


def _round_coord(v: float) -> float:
    # ~0.01 deg (~1km) tak round karo - isse pass-pass ke block centroids
    # same cache-key share kar lete hain agar wo effectively same weather-cell me aate hain.
    return round(v, 2)


def get_archive_weather(lat: float, lon: float, start_date: str, end_date: str) -> Dict:
    """
    Historical weather (hindcast / nowcast reconstruction ke liye).
    start_date, end_date: 'YYYY-MM-DD'
    Returns hourly arrays: time, temperature_2m, relative_humidity_2m,
    precipitation, wind_speed_10m, wind_direction_10m
    """
    lat_r, lon_r = _round_coord(lat), _round_coord(lon)
    key = make_key("archive", lat_r, lon_r, start_date, end_date)

    cached = _cache.get(key, WEATHER_ARCHIVE_TTL)
    if cached is not None:
        return cached

    params = {
        "latitude": lat_r,
        "longitude": lon_r,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _cache.set(key, data, extra_meta={"lat": lat_r, "lon": lon_r})
    return data


def get_forecast_weather(lat: float, lon: float, forecast_days: int = 3) -> Dict:
    """
    Forward-looking weather (forecast mode ke liye - CA projection).
    """
    lat_r, lon_r = _round_coord(lat), _round_coord(lon)
    key = make_key("forecast", lat_r, lon_r, forecast_days)

    cached = _cache.get(key, WEATHER_FORECAST_TTL)
    if cached is not None:
        return cached

    params = {
        "latitude": lat_r,
        "longitude": lon_r,
        "hourly": HOURLY_VARS,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _cache.set(key, data, extra_meta={"lat": lat_r, "lon": lon_r})
    return data


def block_centroids(bbox: Tuple[float, float, float, float], cell_size_deg: float, block_size: int) -> List[Dict]:
    """
    AOI bbox (west, south, east, north) ko block_size x block_size cell-blocks me
    todo aur har block ka centroid lat/lon return karo. Yeh centroids hi
    Open-Meteo query points bante hain - poori AOI ke liye ek-ek pixel query
    karne ki zaroorat nahi.
    """
    west, south, east, north = bbox
    step = cell_size_deg * block_size
    centroids = []
    lat = south
    row = 0
    while lat < north:
        lon = west
        col = 0
        while lon < east:
            centroids.append({
                "row": row, "col": col,
                "lat": lat + step / 2,
                "lon": lon + step / 2,
                "bounds": (lon, lat, min(lon + step, east), min(lat + step, north)),
            })
            lon += step
            col += 1
        lat += step
        row += 1
    return centroids


def fetch_weather_for_grid(bbox, mode: str, cell_size_deg: float, block_size: int,
                            start_date: str = None, end_date: str = None,
                            forecast_days: int = 3) -> List[Dict]:
    """
    Poori AOI ke liye block-wise weather fetch karo (with caching per block).
    mode: 'hindcast' | 'nowcast' -> archive API; 'forecast' -> forecast API
    Returns list of blocks, har ek me centroid info + weather data.
    """
    blocks = block_centroids(bbox, cell_size_deg, block_size)
    results = []
    for b in blocks:
        if mode in ("hindcast", "nowcast"):
            if not (start_date and end_date):
                raise ValueError("hindcast/nowcast ke liye start_date aur end_date chahiye")
            w = get_archive_weather(b["lat"], b["lon"], start_date, end_date)
        elif mode == "forecast":
            w = get_forecast_weather(b["lat"], b["lon"], forecast_days)
        else:
            raise ValueError(f"unknown mode: {mode}")
        results.append({**b, "weather": w})
    return results

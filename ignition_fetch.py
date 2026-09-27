"""
ignition_fetch.py
------------------
FIRMS API se MODIS + VIIRS point-fire detections (lat, lon, timestamp,
confidence, FRP). Same "area" API endpoint, SOURCE parameter switch karke
dono milte hain.

FIRMS_MAP_KEY chahiye - free milti hai:
https://firms.modis.earthdata.nasa.gov/api/map_key/

Ignition seeding logic: FIRMS AOI ke andar jo pehla detection milta hai
(earliest acq_date + acq_time), usko CA grid ka "burning" seed cell maana
jaata hai. FRP ko ignition-intensity proxy ki tarah use karte hain.
"""

import csv
import io
from datetime import datetime
from typing import List, Dict, Tuple

import requests

from config import FIRMS_MAP_KEY, FIRMS_BASE_URL, IGNITION_TTL, IGNITION_CACHE_DIR
from cache_manager import DiskCache, make_key

_cache = DiskCache(IGNITION_CACHE_DIR)

# NRT (near-real-time) sources - nowcast ke liye
SOURCES = {
    "modis": "MODIS_NRT",
    "viirs_snpp": "VIIRS_SNPP_NRT",
    "viirs_noaa20": "VIIRS_NOAA20_NRT",
    # Historical/hindcast ke liye standard (non-NRT) processed archives:
    "modis_std": "MODIS_SP",
    "viirs_snpp_std": "VIIRS_SNPP_SP",
}


def _bbox_str(bbox: Tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return f"{west},{south},{east},{north}"


def fetch_firms_points(bbox: Tuple[float, float, float, float], source: str,
                        day_range: int, end_date: str = None) -> List[Dict]:
    """
    bbox: (west, south, east, north)
    source: key from SOURCES dict, e.g. 'viirs_snpp' or 'modis'
    day_range: kitne din peeche tak dekhna hai (max 10 NRT ke liye)
    end_date: 'YYYY-MM-DD' - is date tak (default = aaj)

    Returns list of dicts: lat, lon, acq_datetime (ISO), confidence, frp
    """
    if not FIRMS_MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY set nahi hai. Environment variable FIRMS_MAP_KEY "
            "set karo (https://firms.modis.earthdata.nasa.gov/api/map_key/ se free milti hai)."
        )

    firms_source = SOURCES.get(source, source)
    key = make_key("firms", bbox, firms_source, day_range, end_date)
    cached = _cache.get(key, IGNITION_TTL)
    if cached is not None:
        return cached

    url_parts = [FIRMS_BASE_URL, "area/csv", FIRMS_MAP_KEY, firms_source, _bbox_str(bbox), str(day_range)]
    if end_date:
        url_parts.append(end_date)
    url = "/".join(url_parts)

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    points = []
    for row in reader:
        try:
            acq_date = row.get("acq_date")
            acq_time = row.get("acq_time", "0000").zfill(4)
            acq_dt = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M")
            points.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "acq_datetime": acq_dt.isoformat(),
                "confidence": row.get("confidence"),
                "frp": float(row["frp"]) if row.get("frp") not in (None, "") else None,
                "satellite": row.get("satellite"),
            })
        except (KeyError, ValueError):
            continue  # malformed row, skip

    points.sort(key=lambda p: p["acq_datetime"])
    _cache.set(key, points, extra_meta={"bbox": bbox, "source": firms_source})
    return points


def find_ignition_seed(bbox: Tuple[float, float, float, float], source: str,
                        day_range: int, end_date: str = None) -> Dict:
    """
    Sabse pehla detection dhoondo AOI+time-window ke andar - yahi CA ka
    ignition seed banega. Nowcast/hindcast dono ke liye common helper.

    NOTE: yeh satellite-overpass-limited hai - exact ignition time nahi,
    balki "jab satellite ne pehli baar fire detect kiya" hai. CA seeding
    isi approximation par based hai.
    """
    points = fetch_firms_points(bbox, source, day_range, end_date)
    if not points:
        return {}
    earliest = points[0]
    return {
        "seed": earliest,
        "all_points_in_window": points,
        "note": "ignition_datetime satellite overpass time hai, exact ignition nahi",
    }

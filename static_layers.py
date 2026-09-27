"""
static_layers.py
------------------
Static/slow conditioning variables, Google Earth Engine se:
  - Curing index  = wet-season NDVI/NDMI baseline vs dry-season reading
  - LAI           = MODIS LAI product
  - Slope/aspect  = SRTM DEM
  - Fuel/landcover = ESA WorldCover (10m, free, global)

Yeh sab "slow" variables hain, isliye har run pe dobara compute nahi karte -
ek baar GeoTIFF me export karke local disk pe cache kar lete hain
(config.STATIC_LAYER_TTL tak valid). Refresh sirf tab jab user explicitly
force karе, ya TTL expire ho jaye (default 30 din).

Setup zaroori hai deploy karte waqt:
    earthengine authenticate     (local/dev ke liye interactive)
    ya production me GEE_SERVICE_ACCOUNT + GEE_PRIVATE_KEY_FILE env vars
"""

from pathlib import Path
from typing import Tuple

import ee

from config import (
    STATIC_LAYERS_DIR, STATIC_LAYER_TTL,
    GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY_FILE,
)
from cache_manager import DiskCache, make_key

_cache = DiskCache(STATIC_LAYERS_DIR)
_ee_initialized = False


def init_ee():
    """GEE ko authenticate/initialize karo. Idempotent - dobara call safe hai."""
    global _ee_initialized
    if _ee_initialized:
        return
    if GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY_FILE:
        credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY_FILE)
        ee.Initialize(credentials)
    else:
        # local/dev: pehle ek baar `earthengine authenticate` chala chuke ho to
        # yeh cached user credentials use kar lega.
        ee.Initialize()
    _ee_initialized = True


def _aoi_geometry(bbox: Tuple[float, float, float, float]) -> "ee.Geometry":
    west, south, east, north = bbox
    return ee.Geometry.Rectangle([west, south, east, north])


def _export_path(name: str, bbox) -> Path:
    key = make_key(name, bbox)
    return _cache.path_for_binary(key, ".tif")


def _is_cached(path: Path) -> bool:
    return _cache.is_fresh(path, STATIC_LAYER_TTL)


def _download_to_geotiff(image: "ee.Image", bbox, out_path: Path, scale: int = 30):
    """geemap ke through GEE image ko local GeoTIFF me export karta hai."""
    import geemap
    region = _aoi_geometry(bbox)
    geemap.ee_export_image(
        image, filename=str(out_path), scale=scale, region=region, file_per_band=False
    )


def get_curing_index(bbox, wet_years: Tuple[int, int] = (2018, 2024),
                      dry_year: int = None, force_refresh: bool = False) -> Path:
    """
    Curing index = seasonal NDVI drop, Sept-Oct (wet/green baseline, multi-year
    climatology) vs next March-April (dry reading).
    Returns: local GeoTIFF path. Values: (NDVI_wet - NDVI_dry) / NDVI_wet, per pixel.
    """
    init_ee()
    out_path = _export_path(f"curing_{wet_years}_{dry_year}", bbox)
    if not force_refresh and _is_cached(out_path):
        return out_path

    aoi = _aoi_geometry(bbox)

    def ndvi(img):
        return img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))

    # Wet baseline: multi-year Sept-Oct median composite (climatology)
    wet_imgs = []
    for yr in range(wet_years[0], wet_years[1] + 1):
        coll = s2.filterDate(f"{yr}-09-01", f"{yr}-10-31")
        wet_imgs.append(coll.map(ndvi).median())
    wet_composite = ee.ImageCollection(wet_imgs).median().rename("NDVI_wet")

    # Dry reading: most recent March-April
    if dry_year is None:
        dry_year = wet_years[1] + 1
    dry_coll = s2.filterDate(f"{dry_year}-03-01", f"{dry_year}-04-30")
    dry_composite = dry_coll.map(ndvi).median().rename("NDVI_dry")

    curing = (wet_composite.subtract(dry_composite)
              .divide(wet_composite.max(0.01))  # avoid div-by-zero
              .rename("curing_index")
              .clamp(0, 1))

    _download_to_geotiff(curing, bbox, out_path, scale=30)
    return out_path


def get_lai(bbox, force_refresh: bool = False) -> Path:
    """MODIS LAI (MCD15A3H, 4-day composite, 500m) - most recent available."""
    init_ee()
    out_path = _export_path("lai", bbox)
    if not force_refresh and _is_cached(out_path):
        return out_path

    aoi = _aoi_geometry(bbox)
    lai_img = (ee.ImageCollection("MODIS/061/MCD15A3H")
               .filterBounds(aoi)
               .sort("system:time_start", False)
               .first()
               .select("Lai")
               .multiply(0.1)  # MODIS scale factor
               .rename("LAI"))

    _download_to_geotiff(lai_img, bbox, out_path, scale=500)
    return out_path


def get_slope_aspect(bbox, force_refresh: bool = False) -> Path:
    """SRTM DEM se slope + aspect (degrees), do-band GeoTIFF."""
    init_ee()
    out_path = _export_path("slope_aspect", bbox)
    if not force_refresh and _is_cached(out_path):
        return out_path

    dem = ee.Image("USGS/SRTMGL1_003")
    terrain = ee.Terrain.products(dem).select(["slope", "aspect"])

    _download_to_geotiff(terrain, bbox, out_path, scale=30)
    return out_path


def get_fuel_landcover(bbox, force_refresh: bool = False) -> Path:
    """ESA WorldCover 10m landcover - fuel-type proxy ke liye."""
    init_ee()
    out_path = _export_path("fuel_landcover", bbox)
    if not force_refresh and _is_cached(out_path):
        return out_path

    lc = ee.ImageCollection("ESA/WorldCover/v200").first().rename("landcover")

    _download_to_geotiff(lc, bbox, out_path, scale=10)
    return out_path


def get_all_static_layers(bbox, force_refresh: bool = False) -> dict:
    """Convenience wrapper - sab static layers ek call me (cached wahi rahenge jo fresh hain)."""
    return {
        "curing_index": str(get_curing_index(bbox, force_refresh=force_refresh)),
        "lai": str(get_lai(bbox, force_refresh=force_refresh)),
        "slope_aspect": str(get_slope_aspect(bbox, force_refresh=force_refresh)),
        "fuel_landcover": str(get_fuel_landcover(bbox, force_refresh=force_refresh)),
    }

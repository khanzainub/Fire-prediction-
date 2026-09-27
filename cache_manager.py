"""
cache_manager.py
-----------------
Ek generic, disk-based cache. Har fetcher (weather, ignition, static layers)
isi ko use karta hai taaki same request baar baar API ko na jaye.

Key idea: har cache-able request ke liye ek deterministic hash key banao
(inputs se), aur uske against JSON/bytes disk par store karo with a
timestamp. Agar timestamp + TTL > abhi ka time, to cached value return karo,
warna None (caller phir fresh fetch karega aur set() call karega).
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


def make_key(*parts: Any) -> str:
    """Deterministic hash key kisi bhi combination of inputs se."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class DiskCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str):
        return self.dir / f"{key}.json", self.dir / f"{key}.meta"

    def get(self, key: str, ttl_seconds: int) -> Optional[Any]:
        data_path, meta_path = self._paths(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return None
        age = time.time() - meta.get("saved_at", 0)
        if age > ttl_seconds:
            return None  # stale - caller will refresh
        try:
            return json.loads(data_path.read_text())
        except Exception:
            return None

    def set(self, key: str, value: Any, extra_meta: Optional[dict] = None) -> None:
        data_path, meta_path = self._paths(key)
        data_path.write_text(json.dumps(value))
        meta = {"saved_at": time.time()}
        if extra_meta:
            meta.update(extra_meta)
        meta_path.write_text(json.dumps(meta))

    def path_for_binary(self, key: str, suffix: str) -> Path:
        """Raster/GeoTIFF jaisi binary files ke liye ek stable path do
        (yeh JSON cache se alag hai - caller khud file likhta/padhta hai)."""
        return self.dir / f"{key}{suffix}"

    def is_fresh(self, path: Path, ttl_seconds: int) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age <= ttl_seconds

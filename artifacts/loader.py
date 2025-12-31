import os
from pathlib import Path
from typing import Any, Dict

import joblib


# Simple in-memory cache keyed by (resolved-path, mtime)
_ARTIFACT_CACHE: Dict[tuple[str, float], Dict[str, Any]] = {}


def _cache_key(p: Path) -> tuple[str, float]:
    """Build a cache key using the resolved path and file mtime.
    If mtime can't be read, use -1.0 to avoid accidental collisions."""
    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = -1.0
    try:
        resolved = str(p.resolve())
    except Exception:
        resolved = str(p)
    return (resolved, mtime)


def load_artifact(path: str) -> Dict[str, Any]:
    """
    Load a saved artifact. Supports joblib-serialized dicts.

    Expected structure (examples):
    - Prophet:
      {"model_type": "prophet", "model": <Prophet>, "feature_columns": [...], "metrics": {...}}
    - Hybrid Prophet+XGBoost:
      {"model_type": "hybrid", "prophet": <Prophet>, "xgb": <XGBRegressor>, "feature_columns": [...], "metrics": {...}}

    Returns the artifact dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    # Return from cache if present and file unchanged
    key = _cache_key(p)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached

    # Only joblib for now to match app.py expectations
    try:
        obj = joblib.load(str(p))
    except Exception as e:
        raise RuntimeError(f"Failed to load artifact via joblib at {path}: {e}")

    if not isinstance(obj, dict) or "model_type" not in obj:
        # Keep backward compatibility: wrap raw model objects if needed
        wrapped = {}
        if obj.__class__.__name__ == "Prophet":
            wrapped = {"model_type": "prophet", "model": obj}
        else:
            wrapped = {"model_type": "unknown", "model": obj}
        _ARTIFACT_CACHE[key] = wrapped
        # Purge older versions of same path (different mtime)
        for k in list(_ARTIFACT_CACHE.keys()):
            if k[0] == key[0] and k != key:
                _ARTIFACT_CACHE.pop(k, None)
        return wrapped

    # Cache the successfully loaded dict artifact
    _ARTIFACT_CACHE[key] = obj
    # Purge older versions of same path (different mtime)
    for k in list(_ARTIFACT_CACHE.keys()):
        if k[0] == key[0] and k != key:
            _ARTIFACT_CACHE.pop(k, None)
    return obj

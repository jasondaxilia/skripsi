import os
from pathlib import Path
from typing import Any, Dict

import joblib


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
        return wrapped

    return obj

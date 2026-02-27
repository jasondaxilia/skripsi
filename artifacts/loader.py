import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pickle


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

    # Detect Git LFS pointer files and guide the user
    try:
        if p.is_file() and p.stat().st_size < 2048:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(256)
            if head.startswith("version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    (
                        "The artifact at {} is a Git LFS pointer, not the actual binary. "
                        "Fetch LFS objects (e.g., 'git lfs pull') or use the real artifact path."
                    ).format(path)
                )
    except Exception:
        # Non-fatal: continue to loader fallbacks
        pass

    # Return from cache if present and file unchanged
    key = _cache_key(p)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached

    # Try multiple strategies to load the artifact for robustness
    try:
        obj = joblib.load(str(p))
    except Exception as e1:
        # Try memory-mapped mode which can help with large numpy arrays
        try:
            obj = joblib.load(str(p), mmap_mode="r")
        except Exception as e2:
            # Fallback to standard pickle in case the file isn't a joblib archive
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
            except Exception as e3:
                raise RuntimeError(
                    "Failed to load artifact at {}. "
                    "joblib: {} ({}); joblib(mmap): {} ({}); pickle: {} ({})".format(
                        path,
                        type(e1).__name__, e1,
                        type(e2).__name__, e2,
                        type(e3).__name__, e3,
                    )
                )

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

    # === CRITICAL FIX: Load NeuralProphet from saved path if model_type is neuralprophet ===
    # NeuralProphet often can't be pickled directly, so notebook saves it via model.save()
    # and stores the path in 'model_dir' key
    if obj.get("model_type") == "neuralprophet":
        # Check if model was saved via NeuralProphet.save() method
        if "model_dir" in obj and "neuralprophet" not in obj:
            model_dir = obj.get("model_dir")
            try:
                from neuralprophet import NeuralProphet
                # Load model from saved directory
                neural_model = NeuralProphet.load(model_dir)
                obj["neuralprophet"] = neural_model
                print(f"✅ Loaded NeuralProphet from: {model_dir}")
            except Exception as e:
                import streamlit as st
                st.warning(f"Failed to load NeuralProphet from {model_dir}: {e}")
                # Don't cache this incomplete artifact
                return obj
    
    # === CRITICAL FIX: Load N-BEATS PyTorch model from .pth file if needed ===
    # N-BEATS saves state dict to separate .pth file (model_state_path)
    if obj.get("model_type") == "nbeats":
        # Check if model is in separate .pth file (new format)
        if "model_state_path" in obj and "model_state_dict" not in obj:
            model_state_path = obj.get("model_state_path")
            try:
                import torch
                # Resolve path relative to repository root
                pth_path = Path(model_state_path)
                if not pth_path.is_absolute():
                    # Try relative to current artifact location
                    pth_path = p.parent / model_state_path
                    if not pth_path.exists():
                        # Try from repo root
                        pth_path = Path(model_state_path)
                
                # Load state dict from .pth file
                state_dict = torch.load(str(pth_path), map_location='cpu')
                obj["model_state_dict"] = state_dict
                print(f"✅ Loaded N-BEATS state dict from: {pth_path}")
            except Exception as e:
                print(f"⚠️  Failed to load N-BEATS from {model_state_path}: {e}")
                # Continue anyway - predictors.py will handle fallback

    # Cache the successfully loaded dict artifact
    _ARTIFACT_CACHE[key] = obj
    # Purge older versions of same path (different mtime)
    for k in list(_ARTIFACT_CACHE.keys()):
        if k[0] == key[0] and k != key:
            _ARTIFACT_CACHE.pop(k, None)
    return obj


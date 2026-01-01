from __future__ import annotations

from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import logging as logging
from darts.models import NHiTSModel

def _np_expected_regressors(m) -> List[str]:
    """Best-effort introspection of NeuralProphet model to discover expected regressors.
    Supports multiple NP versions by checking several config locations.
    """
    names: List[str] = []
    cfg = None
    try:
        cfg = getattr(m, "config_train", None) or getattr(m, "config", None)
    except Exception:
        cfg = None

    candidates = []
    # Common cases
    try:
        reg = getattr(cfg, "regressors", None)
        if reg is not None:
            if isinstance(reg, dict):
                candidates.extend(list(reg.keys()))
            elif isinstance(reg, (list, tuple)):
                candidates.extend([str(x) for x in reg])
    except Exception:
        pass
    # Other possible config fields across versions
    for attr in [
        "regressors_config",
        "config_regressors",
        "future_regressors",
    ]:
        try:
            obj = getattr(cfg if cfg is not None else m, attr, None)
            if obj is None:
                continue
            if isinstance(obj, dict):
                candidates.extend(list(obj.keys()))
            elif isinstance(obj, (list, tuple)):
                candidates.extend([str(x) for x in obj])
        except Exception:
            pass

    # Deduplicate and sort for stability
    names = sorted(set([c for c in candidates if isinstance(c, str) and c]))
    return names

def _infer_freq_from_ds(ds: pd.Series) -> str | None:
    try:
        freq = pd.infer_freq(pd.to_datetime(ds))
        if freq:
            return freq
    except Exception:
        pass
    # Fallback: assume business daily if dates look daily
    return "B"


def _future_frame_from_last(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Create a naive future frame by extending dates daily and forward-filling features.
    Assumes df has a 'ds' column and feature columns to be carried forward.
    """
    if "ds" not in df.columns:
        raise ValueError("Dataframe must contain 'ds' column for future generation.")

    last_date = pd.to_datetime(df["ds"].iloc[-1])
    freq = _infer_freq_from_ds(df["ds"]) or "D"
    # Advance by one step in inferred frequency
    try:
        future_dates = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    except Exception:
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
    # start from the last known row and forward-fill features constant into future
    last_row = df.iloc[-1:].copy()
    future = pd.DataFrame({"ds": future_dates})
    # carry-forward all non-target numeric features
    feature_cols = [c for c in df.columns if c not in {"y"}]
    for c in feature_cols:
        if c == "ds":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            future[c] = last_row[c].values[0]
        else:
            # leave non-numeric features empty if any
            future[c] = None
    return future


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    # passthrough to features.ensure_schema to avoid circular import; redefine for app import stability
    from .features import ensure_schema as _ensure
    return _ensure(df)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # passthrough to features.build_features to avoid circular import; redefine for app import stability
    from .features import build_features as _build
    return _build(df)


def _predict_prophet(m, df: pd.DataFrame, feature_cols: List[str] | None, periods: int, scaler=None) -> pd.DataFrame:
    # Build future; if feature_cols required, ensure present
    future = _future_frame_from_last(df, periods)
    if feature_cols:
        missing = [c for c in feature_cols if c not in future.columns]
        if missing:
            # add missing regressors with last-known values if possible
            for c in missing:
                future[c] = df[c].iloc[-1] if c in df.columns else 0.0
        # Apply scaler if provided to match training-time preprocessing
        if scaler is not None:
            try:
                Xf = future[feature_cols].values
                Xf_scaled = scaler.transform(Xf)
                future.loc[:, feature_cols] = Xf_scaled
            except Exception:
                pass
    fc = m.predict(future)
    return fc[["ds", "yhat"]]


def _predict_neuralprophet(m, df: pd.DataFrame, feature_cols: List[str] | None, periods: int, scaler=None) -> pd.DataFrame:
    """Predict with NeuralProphet, ensuring regressors are present and scaled.
    Returns a dataframe with columns: ds, yhat
    """
    df_in = df.copy()

    # Ensure target column 'y' exists (NeuralProphet expects 'ds' + 'y')
    if "y" not in df_in.columns:
        if "Close" in df_in.columns:
            df_in["y"] = df_in["Close"]
        else:
            raise ValueError("NeuralProphet prediction requires 'y' (or 'Close') column in input dataframe.")

    # Determine expected regressors from model; fall back to provided feature_cols
    expected_regs = _np_expected_regressors(m)
    use_feature_cols = feature_cols or []
    if expected_regs:
        # Align to model-expected names only
        use_feature_cols = expected_regs

    # Ensure required feature columns exist; fill from last known or 0.0
    if use_feature_cols:
        missing_hist = [c for c in use_feature_cols if c not in df_in.columns]
        for c in missing_hist:
            df_in[c] = 0.0
        # Apply scaler to history features to match training preprocessing
        if scaler is not None:
            try:
                Xh = df_in[use_feature_cols].values
                Xh_scaled = scaler.transform(Xh)
                df_in.loc[:, use_feature_cols] = Xh_scaled
            except Exception:
                pass

    # Drop raw 'Close' to avoid unexpected column errors in NP
    if "Close" in df_in.columns:
        try:
            df_in = df_in.drop(columns=["Close"])
        except Exception:
            pass

    # Build future frame naively (robust across NP versions)
    future = _future_frame_from_last(df_in, periods)
    # Ensure future has placeholder target column to satisfy versions expecting 'y'
    if "y" not in future.columns:
        future["y"] = np.nan

    # Ensure future has all feature columns, carry forward last known values
    if use_feature_cols:
        for c in use_feature_cols:
            if c not in future.columns:
                future[c] = df_in[c].iloc[-1] if c in df_in.columns else 0.0
        # Drop any extra columns not expected by the model
        extra_cols = [c for c in future.columns if c not in set(["ds", "y"]) | set(use_feature_cols)]
        if extra_cols:
            try:
                future = future.drop(columns=extra_cols)
            except Exception:
                pass
        # Scale future feature values if scaler provided
        if scaler is not None:
            try:
                Xf = future[use_feature_cols].values
                Xf_scaled = scaler.transform(Xf)
                future.loc[:, use_feature_cols] = Xf_scaled
            except Exception:
                pass

    # Restrict to allowed columns only: ds, y, and declared regressors
    use_cols = [c for c in ["ds", "y"] if c in future.columns] + (use_feature_cols or [])
    # Drop any unexpected columns (e.g., 'Close')
    try:
        future = future[use_cols]
    except Exception:
        # If subsetting fails, at least drop 'Close' explicitly
        if "Close" in future.columns:
            future = future.drop(columns=["Close"])

    # Try prediction; retry with stricter columns if mismatch errors arise
    try:
        fc = m.predict(future)
    except Exception as e:
        msg = str(e)
        # Some NP versions error if 'y' is missing in future
        if isinstance(e, KeyError) and str(e) == "'y'" and "y" not in future.columns:
            future["y"] = np.nan
            fc = m.predict(future)
        # Handle mismatch error: limits vs iterables (often due to regressor count mismatch)
        elif "number of limits" in msg and "number of iterables" in msg:
            try:
                # Restrict to ds + expected regressors only; drop 'y' for strict future
                cols_retry = ["ds"] + (use_feature_cols or [])
                future_retry = future[cols_retry].copy()
                fc = m.predict(future_retry)
            except Exception:
                # Final fallback: include 'y' as NaN and retry
                try:
                    future_retry["y"] = np.nan
                    fc = m.predict(future_retry)
                except Exception:
                    raise
        else:
            raise
    # Normalize output column to 'yhat'
    if "yhat" in fc.columns:
        out = fc[["ds", "yhat"]].copy()
    elif "yhat1" in fc.columns:
        out = fc[["ds", "yhat1"]].copy()
        out = out.rename(columns={"yhat1": "yhat"})
    else:
        yhat_col = next((c for c in fc.columns if c.startswith("yhat")), None)
        if yhat_col is None:
            raise ValueError("NeuralProphet prediction output missing 'yhat' columns.")
        out = fc[["ds", yhat_col]].copy().rename(columns={yhat_col: "yhat"})
    return out


def predict_model(artifact: Dict[str, Any], df: pd.DataFrame, periods: int, debug: bool = False) -> pd.DataFrame:
    """Run prediction for a given artifact and feature-augmented dataframe.
    Returns a tidy dataframe with columns: ds, yhat, model
    """
    model_type = artifact.get("model_type", "unknown")

    if model_type == "prophet":
        m = artifact.get("model") or artifact.get("prophet")
        if m is None:
            raise ValueError("Prophet artifact missing 'model'.")
        feature_cols = artifact.get("feature_columns")
        scaler = artifact.get("scaler")
        out = _predict_prophet(m, df, feature_cols, periods, scaler=scaler)
        out["model"] = "Prophet"
        if debug:
            artifact["_debug_future"] = out
        return out

    if model_type == "hybrid":
        m = artifact.get("prophet")
        xgb = artifact.get("xgb")
        feature_cols = artifact.get("feature_columns") or []
        scaler = artifact.get("scaler")
        if m is None or xgb is None:
            raise ValueError("Hybrid artifact must contain 'prophet' and 'xgb'.")

        base = _predict_prophet(m, df, feature_cols, periods, scaler=scaler)

        future = _future_frame_from_last(df, periods)
        Xf = future[feature_cols].values if feature_cols else np.empty((len(future), 0))
        if scaler is not None and Xf.size > 0:
            try:
                Xf = scaler.transform(Xf)
                future.loc[:, feature_cols] = Xf
            except Exception:
                pass
        try:
            residual = xgb.predict(Xf) if Xf.size > 0 else np.zeros(len(future))
        except Exception:
            residual = np.zeros(len(future))

        out = base.copy()
        out["yhat"] = out["yhat"].values + residual
        out["model"] = "Hybrid"
        if debug:
            artifact["_debug_future"] = future
            artifact["_debug_base"] = base
        return out

    if model_type == "nhits":
        nhits_path = artifact.get("nhits_path")
        if not nhits_path:
            raise ValueError("NHITS artifact missing 'nhits_path' (.darts file).")

        m = NHiTSModel.load(nhits_path)
        if m is None:
            raise ValueError("NHITS artifact missing 'model'.")

        scaler_y = artifact.get("scaler_y")
        scaler_cov = artifact.get("scaler_cov")
        
        target_col = "y" if "y" in df.columns else ("Close" if "Close" in df.columns else None)
        if target_col is None:
            raise ValueError("NHITS prediction requires either 'y' or 'Close' column as target.")
        feature_cols = artifact.get("feature_columns") or [
            c for c in df.columns if c not in {"ds", "y", "Close"}
        ]
        # Debug: check feature alignment and numeric types
        try:
            missing_features = [c for c in feature_cols if c not in df.columns]
            non_numeric = [c for c in feature_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
            nan_counts = {c: int(pd.to_numeric(df[c], errors="coerce").isna().sum()) for c in feature_cols if c in df.columns}
            artifact["_debug_nhits_mismatch"] = {
                "missing_features": missing_features,
                "non_numeric_features": non_numeric,
                "nan_counts": nan_counts,
            }
        except Exception:
            pass

        from darts import TimeSeries
        from darts.dataprocessing.transformers import Scaler as DartsScaler

        if "ds" not in df.columns:
            raise ValueError("NHITS prediction requires dataframe with 'ds' column.")

        # Mirror notebook preprocessing to avoid NaNs
        inferred_freq = "D"
        df_prophet = df.rename(columns={"Close": "y"}) if "Close" in df.columns else df.copy()
        if "y" not in df_prophet.columns:
            raise ValueError("NHITS requires 'Close' or 'y' column for target.")
        # Build model-ready dataframe using selected features
        cols_for_model = ["ds", "y"] + feature_cols
        df_model_ready = df_prophet[cols_for_model].copy()
        df_model_ready["ds"] = pd.to_datetime(df_model_ready["ds"])  # ensure datetime
        # Ensure daily frequency and forward-fill
        df_model_ready = df_model_ready.set_index("ds").asfreq(inferred_freq).ffill().reset_index()
        # Coerce numerics and sanitize infinities before dropping NaNs
        df_model_ready["y"] = pd.to_numeric(df_model_ready["y"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for c in feature_cols:
            df_model_ready[c] = pd.to_numeric(df_model_ready[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Drop rows with missing target or features (matches notebook behavior)
        df_model_ready = df_model_ready.dropna(subset=["y"] + feature_cols).reset_index(drop=True)

        # Target series
        try:
            ts_y = TimeSeries.from_dataframe(df_model_ready, "ds", "y", fill_missing_dates=True, freq=inferred_freq)
        except Exception:
            ts_y = TimeSeries.from_dataframe(df_model_ready, "ds", "y", freq=inferred_freq)
        try:
            ts_y = ts_y.fill_missing_values()
        except Exception:
            pass

        # Covariates: use full cleaned history and extend into future using last values
        ts_cov = None
        if feature_cols:
            cov_hist = df_model_ready[["ds"] + feature_cols].copy()
            last_date = pd.to_datetime(df_model_ready["ds"].iloc[-1])
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq=inferred_freq)
            cov_future = pd.DataFrame({"ds": future_dates})
            for c in feature_cols:
                cov_future[c] = cov_hist[c].iloc[-1]
            cov_ext = pd.concat([cov_hist, cov_future], ignore_index=True)
            try:
                ts_cov = TimeSeries.from_dataframe(cov_ext, "ds", feature_cols, fill_missing_dates=True, freq=inferred_freq)
            except Exception:
                ts_cov = TimeSeries.from_dataframe(cov_ext, "ds", feature_cols, freq=inferred_freq)
            try:
                ts_cov = ts_cov.fill_missing_values()
            except Exception:
                pass

        # Apply saved scalers when available
        ts_y_s = ts_y
        ts_cov_s = ts_cov
        try:
            if scaler_y is not None and isinstance(scaler_y, DartsScaler):
                ts_y_s = scaler_y.transform(ts_y)
        except Exception:
            pass
        try:
            if ts_cov is not None and scaler_cov is not None and isinstance(scaler_cov, DartsScaler):
                ts_cov_s = scaler_cov.transform(ts_cov)
            else:
                ts_cov_s = ts_cov
        except Exception:
            ts_cov_s = ts_cov

        # Collect NHITS debug info for display in app
        try:
            info = {
                "feature_columns": feature_cols,
                "freq": inferred_freq,
                "ts_y_len": int(len(ts_y)) if ts_y is not None else 0,
                "ts_cov_len": int(len(ts_cov)) if ts_cov is not None else 0,
                "ts_y_nan": int(ts_y.pd_dataframe().isna().sum().sum()) if ts_y is not None else None,
                "ts_cov_nan": ts_cov.pd_dataframe().isna().sum().to_dict() if ts_cov is not None else None,
                "scaler_y": getattr(getattr(scaler_y, "scaler", None), "__class__", type(None)).__name__ if scaler_y is not None else None,
                "scaler_cov": getattr(getattr(scaler_cov, "scaler", None), "__class__", type(None)).__name__ if scaler_cov is not None else None,
                "input_chunk_length": getattr(m, "input_chunk_length", None),
                "output_chunk_length": getattr(m, "output_chunk_length", None),
            }
            artifact["_debug_nhits_info"] = info
        except Exception:
            pass

        # If the model was trained with covariates, make sure we provide them
        if ts_cov_s is None:
            raise ValueError("NHITS requires past covariates but none were provided. Ensure feature_columns are present and numeric.")

        # Predict next N steps
        forecast_s = m.predict(n=periods, series=ts_y_s, past_covariates=ts_cov_s)
        # Fallback: if prediction is entirely NaN, try without scaling
        try:
            vals_chk = forecast_s.values().flatten()
            if not np.isfinite(np.nanmax(vals_chk)):
                # Retry with unscaled and sanitized inputs
                forecast_s = m.predict(n=periods, series=ts_y, past_covariates=ts_cov)
        except Exception:
            pass
        # Inverse-transform to original scale if scaler_y provided
        try:
            if scaler_y is not None and isinstance(scaler_y, DartsScaler):
                forecast = scaler_y.inverse_transform(forecast_s)
            else:
                forecast = forecast_s
        except Exception:
            forecast = forecast_s

        # If inverse-transformed forecast is still NaN, fall back to persistence (last known Close)
        try:
            vals_chk = forecast.values().flatten()
            has_finite = np.isfinite(np.nanmax(vals_chk))
        except Exception:
            has_finite = False
        if not has_finite:
            last_close = None
            try:
                # Prefer original price scale from input df
                last_close = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
                last_close = float(last_close.iloc[-1]) if len(last_close) else None
            except Exception:
                last_close = None
            if last_close is None:
                try:
                    last_close = float(ts_y.last_value())
                except Exception:
                    last_close = None
            if last_close is not None and np.isfinite(last_close):
                # Build a simple persistence forecast aligned to inferred frequency
                from darts import TimeSeries as _TS
                last_date = pd.to_datetime(df["ds"].iloc[-1])
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
                fallback_df = pd.DataFrame({"ds": future_dates, "y": [last_close] * periods})
                try:
                    forecast = _TS.from_dataframe(fallback_df, "ds", "y", fill_missing_dates=True, freq=inferred_freq)
                except Exception:
                    forecast = _TS.from_dataframe(fallback_df, "ds", "y", freq=inferred_freq)
                if debug:
                    artifact["_debug_nhits_fallback"] = {
                        "reason": "All-NaN forecast; using persistence fallback",
                        "last_close": last_close,
                    }

        # Build output dataframe with future dates
        last_date = pd.to_datetime(df["ds"].iloc[-1])
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
        # Extract numeric forecast values robustly
        yhat_raw = forecast.values().flatten()
        try:
            yhat_num = pd.to_numeric(pd.Series(yhat_raw), errors="coerce").astype(float).values
        except Exception:
            yhat_num = np.array(yhat_raw, dtype=float)
        # Align lengths just in case
        min_len = min(len(future_dates), len(yhat_num))
        out = pd.DataFrame({
            "ds": future_dates[:min_len],
            "yhat": yhat_num[:min_len],
        })
        # Align label with app expectations
        out["model"] = "NHITS"
        if debug:
            artifact["_debug_series_y"] = ts_y
            artifact["_debug_series_cov"] = ts_cov
        return out
      
    if model_type == "neuralprophet":
        m = artifact.get("model") or artifact.get("neuralprophet")
        if m is None:
            raise ValueError("NeuralProphet artifact missing 'model'.")
        feature_cols = artifact.get("feature_columns")
        scaler = artifact.get("scaler")
        # Debug: record expected regressors from the model
        try:
            artifact["_debug_np_expected_regressors"] = _np_expected_regressors(m)
        except Exception:
            pass
        out = _predict_neuralprophet(m, df, feature_cols or [], periods, scaler=scaler)
        out["model"] = "NeuralProphet"
        if debug:
            artifact["_debug_future"] = out
        return out
    # Unknown model: no-op
    raise ValueError(f"Unsupported model_type: {model_type}")

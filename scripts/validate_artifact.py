import argparse
from pathlib import Path
import sys
import json
import warnings
import os

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure workspace root is on sys.path so 'artifacts' package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports from workspace
try:
    from artifacts.loader import load_artifact
    from artifacts.features import ensure_schema, build_features
    from artifacts.predictors import predict_model
except Exception as e:
    print(f"Failed to import local modules: {e}")
    print(f"Added workspace root to sys.path: {ROOT}")
    print("Ensure you run this script from the workspace root.")
    sys.exit(1)

warnings.filterwarnings("ignore")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    if len(dy_true) == 0 or len(dy_pred) == 0:
        return np.nan
    return float(np.mean(np.sign(dy_true) == np.sign(dy_pred)))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = pd.to_numeric(y_true, errors="coerce").dropna()
    y_pred = pd.to_numeric(y_pred, errors="coerce").reindex(y_true.index).dropna()
    # align indices
    common = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common]
    y_pred = y_pred.loc[common]
    if len(common) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan, "DirAcc": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) if np.all(y_true != 0) else np.nan
    r2 = r2_score(y_true, y_pred)
    dir_acc = directional_accuracy(y_true.values, y_pred.values)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "DirAcc": dir_acc}


def load_metrics_from_artifact(path: Path) -> dict:
    try:
        art = load_artifact(str(path))
        return art.get("metrics", {}) or {}
    except Exception:
        return {}


def validate_artifacts(ticker: str, years: int, horizon: int, models: dict) -> None:
    print(f"Downloading {years}y of daily data for {ticker}...")
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    df = df.reset_index()
    df = flatten_columns(df)

    # Ensure schema and features
    df = ensure_schema(df)
    feat = build_features(df)
    if feat is None or feat.empty:
        print("Feature building failed; check data columns and history length.")
        sys.exit(2)

    # Base series
    close = feat[["ds", "Close"]].set_index("ds")["Close"]

    # App-style predictions using shared predictor
    app_preds = {}
    artifact_info = {}
    for name, path in models.items():
        p = Path(path)
        if not p.exists():
            print(f"Warning: missing artifact for {name}: {path}")
            continue
        try:
            artifact = load_artifact(str(p))
            mt = (artifact.get("model_type") or "").lower()
            has_scaler = False
            if mt == "nhits":
                has_scaler = bool(artifact.get("scaler_y")) and bool(artifact.get("scaler_cov"))
            else:
                has_scaler = bool(artifact.get("scaler"))
            artifact_info[name] = {
                "path": str(p),
                "keys": list(artifact.keys()),
                "model_type": artifact.get("model_type"),
                "has_scaler": has_scaler,
                "feature_columns": artifact.get("feature_columns") or [],
            }
            pred = predict_model(artifact, feat, horizon, debug=False)
            dfp = pred.set_index("ds")["yhat"].rename(name)
            app_preds[name] = dfp
        except Exception as e:
            print(f"Skipping {name} app-style prediction: {e}")

    # Compute metrics where possible (historical in-sample)
    # For Prophet, we can compare the model's in-sample yhat where available
    metrics = {}
    for name, series in app_preds.items():
        # Only evaluate if overlap with history
        overlap = series.index.intersection(close.index)
        if len(overlap) > 0:
            m = compute_metrics(close.loc[overlap], series.loc[overlap])
            metrics[name] = m

    print("\n=== App-Style Predictions (last horizon days) ===")
    for name, series in app_preds.items():
        tail = series.tail(horizon)
        print(f"\n{name} forecast:")
        print(tail.to_frame())

    print("\n=== Metrics vs Close (overlapping indices) ===")
    for name, m in metrics.items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, (int, float)) and not np.isnan(v) else f"  {k}: {v}")

    # Compare stored metrics in artifacts if present
    print("\n=== Stored Artifact Metrics (from training) ===")
    for name, path in models.items():
        m = load_metrics_from_artifact(Path(path))
        print(f"\n{name}:")
        if m:
            try:
                print(json.dumps(m, indent=2))
            except Exception:
                print(m)
        else:
            print("(no metrics found)")

    # Show artifact details (presence of scaler, feature columns)
    print("\n=== Artifact Details ===")
    for name, info in artifact_info.items():
        print(f"\n{name} ({info['path']}):")
        print(f"  model_type: {info['model_type']}")
        print(f"  has_scaler: {info['has_scaler']}")
        print(f"  feature_columns: {len(info['feature_columns'])} columns")
        if info["feature_columns"]:
            preview = info["feature_columns"][:10]
            print(f"    preview: {preview}")

    # Sanity check: scale vs last close
    last_close = float(close.iloc[-1])
    print("\n=== Scale Sanity Check ===")
    for name, series in app_preds.items():
        last_pred = float(series.dropna().iloc[-1]) if len(series.dropna()) else np.nan
        ratio = last_pred / last_close if np.isfinite(last_pred) and last_close != 0 else np.nan
        print(f"{name}: last_pred={last_pred:.6f} last_close={last_close:.6f} ratio={ratio:.6f}")

    # Guidance if scaler is missing
    if any((not artifact_info.get(n, {}).get("has_scaler", False)) for n in artifact_info):
        print("\n=== Action Required: Re-export Artifacts with Scaler ===")
        print("Your artifacts appear to be missing the fitted scaler used during training.")
        print("This causes predictions at unrealistic scales (e.g., 1e12).")
        print("Please re-run your training notebook and save artifacts including the scaler and feature columns.")
        print("\nProphet/Hybrid example (MinMaxScaler on regressors):\n")
        print(
            """
from pathlib import Path
import joblib

# After fitting scaler, Prophet, and XGBoost (hybrid)
export_dir = Path("artifacts/models")
export_dir.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model_type": "prophet",
    "prophet": prophet_model,  # fitted Prophet
    "scaler": fitted_scaler,   # MinMaxScaler fitted on training regressors
    "feature_columns": feature_cols,  # list of regressor column names
    "metrics": prophet_metrics,
}, export_dir / "prophet.joblib")

joblib.dump({
    "model_type": "hybrid",
    "prophet": prophet_model,
    "xgb": xgb_model,          # fitted XGBoost on residuals
    "scaler": fitted_scaler,
    "feature_columns": feature_cols,
    "metrics": hybrid_metrics,
}, export_dir / "hybrid.joblib")
            """
        )
        print("\nNHITS example (Darts Scaler on target and covariates):\n")
        print(
            """
from pathlib import Path
import joblib
from darts.models import NHiTSModel

export_dir = Path("models")
export_dir.mkdir(parents=True, exist_ok=True)

# Assume you've trained NHiTS as `model`, and fitted Darts `scaler_y` & `scaler_cov`.
nhits_path = export_dir / "DEWA_nhits.darts"  # adjust ticker
model.save(str(nhits_path))

joblib.dump({
    "model_type": "nhits",
    "nhits_path": str(nhits_path.resolve()),
    "scaler_y": scaler_y,
    "scaler_cov": scaler_cov,
    "feature_columns": feature_cols,
    "metrics": nhits_metrics,
}, export_dir / "DEWA_nhits.joblib")
            """
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Prophet/Hybrid artifacts against app-style predictions and show metrics.")
    parser.add_argument("--ticker", default="BUMI.JK", help="Ticker symbol, e.g., BUMI.JK")
    parser.add_argument("--years", type=int, default=2, help="Years of history to download for validation")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (days)")
    parser.add_argument("--prophet", default="models/prophet.joblib", help="Path to Prophet artifact")
    parser.add_argument("--hybrid", default="models/hybrid.joblib", help="Path to Hybrid artifact")
    args = parser.parse_args()

    models = {
        "Prophet": args.prophet,
        "Hybrid": args.hybrid,
    }
    validate_artifacts(args.ticker, args.years, args.horizon, models)

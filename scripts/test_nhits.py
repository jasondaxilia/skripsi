import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ensure workspace root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.loader import load_artifact
from artifacts.features import ensure_schema, build_features
from artifacts.predictors import predict_model
from darts.models import NHiTSModel
import os
import datetime as dt


def main(ticker: str = "BUMI.JK", horizon: int = 5, artifact_path: str | None = None):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = ensure_schema(df)
    feat = build_features(df)

    if artifact_path is None:
        # default inferred path from ticker symbol
        emiten = ticker.split(".")[0]
        artifact_path = f"models/{emiten}_nhits.joblib"

    print(f"Loading NHITS artifact: {artifact_path}")
    art = load_artifact(artifact_path)
    print("Artifact keys:", list(art.keys()))

    # Inspect the saved NHITS model directly to confirm config matches training
    nhits_path = art.get("nhits_path")
    if nhits_path and os.path.exists(nhits_path):
        try:
            model = NHiTSModel.load(nhits_path)
            ocl = getattr(model, "output_chunk_length", None)
            icl = getattr(model, "input_chunk_length", None)
            mtime = dt.datetime.fromtimestamp(os.path.getmtime(nhits_path))
            print("\n=== Loaded NHITS Model Info ===")
            print(f"nhits_path: {nhits_path}")
            print(f"modified: {mtime}")
            print(f"input_chunk_length: {icl}")
            print(f"output_chunk_length: {ocl}")
        except Exception as e:
            print("\n⚠️ Could not load NHITS model for inspection:", e)
    else:
        print("\n⚠️ nhits_path missing or file not found in artifact.")

    pred = predict_model(art, feat, horizon, debug=True)
    print("Prediction head:")
    print(pred)
    print("Value types:")
    print(pred.dtypes)
    # Print NHITS debug info and fallback state
    info = art.get("_debug_nhits_info")
    if info:
        print("\n=== NHITS Debug Info ===")
        for k, v in info.items():
            print(f"{k}: {v}")
    fb = art.get("_debug_nhits_fallback")
    if fb:
        print("\n=== NHITS Fallback ===")
        print(fb)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="BUMI.JK")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    main(args.ticker, args.horizon, args.artifact)

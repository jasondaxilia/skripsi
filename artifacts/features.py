from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import logging as log

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has at least columns: ds (datetime), Close.
    - If there's a 'Date' or other datetime-like column, rename to 'ds'.
    - If no datetime column exists but index is datetime-like, move index to 'ds'.
    - Converts ds to pandas datetime, sorts, dedupes.
    """
    out = df.copy()

    # Try to locate a date column
    date_candidates = [
        "ds", "Date", "date", "Datetime", "datetime", "Timestamp", "timestamp", "index"
    ]

    found_col = None
    for col in date_candidates:
        if col in out.columns:
            found_col = col
            break

    # If not found, but index is datetime-like, use it
    if found_col is None:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            found_col = "index"

    # Rename to ds if needed
    if found_col and found_col != "ds":
        out = out.rename(columns={found_col: "ds"})

    # If still missing, try to detect any datetime-like column by dtype or parsability
    if "ds" not in out.columns:
        for c in list(out.columns):
            try:
                # attempt lightweight parse on a sample
                sample = out[c].iloc[:5]
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().any():
                    out = out.rename(columns={c: "ds"})
                    break
            except Exception:
                continue

    # Final check
    if "ds" not in out.columns:
        raise ValueError("Input data must contain a datetime column (e.g., 'Date') or have a DatetimeIndex.")
    if "Close" not in out.columns:
        # Try common alternative
        if "Adj Close" in out.columns:
            out = out.rename(columns={"Adj Close": "Close"})
        else:
            raise ValueError("Input data must contain a 'Close' price column.")

    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds"]).sort_values("ds").drop_duplicates(subset=["ds"])  # stable time order
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators required by the hybrid notebook.
    Adds the following columns where possible:
    - Prev Close, MA20, MA50, MACD, Signal Line,
      Lag1..Lag5, RSI, BB_Upper, BB_Lower, ATR,
      Stochastic_K, Stochastic_D, CCI, OBV, Volume

    Returns a new dataframe with added features.
    """
    out = df.copy()
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    if "High" not in out.columns:
        out["High"] = out["Close"]
    if "Low" not in out.columns:
        out["Low"] = out["Close"]

    # Prev Close
    out["Prev Close"] = out["Close"].shift(1)

    # Moving averages
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()

    # MACD and Signal line
    ema12 = _ema(out["Close"], 12)
    ema26 = _ema(out["Close"], 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["MACD"] = macd
    out["Signal Line"] = signal

    # Lags
    for k in range(1, 6):
        out[f"Lag{k}"] = out["Close"].shift(k)

    # RSI
    out["RSI"] = _rsi(out["Close"], 14)

    # Bollinger Bands (20-day)
    rolling_mean = out["Close"].rolling(20, min_periods=20).mean()
    rolling_std = out["Close"].rolling(20, min_periods=20).std()
    out["BB_Upper"] = rolling_mean + 2 * rolling_std
    out["BB_Lower"] = rolling_mean - 2 * rolling_std

    # ATR
    out["ATR"] = _atr(out["High"], out["Low"], out["Close"], 14)

    # Stochastic Oscillator (14)
    ll = out["Low"].rolling(14, min_periods=14).min()
    hh = out["High"].rolling(14, min_periods=14).max()
    k = (out["Close"] - ll) / (hh - ll).replace(0, np.nan) * 100
    out["Stochastic_K"] = k
    out["Stochastic_D"] = k.rolling(3, min_periods=3).mean()

    # CCI (20)
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    sma_typ = typical.rolling(20, min_periods=20).mean()
    mad = (typical - sma_typ).abs().rolling(20, min_periods=20).mean()
    out["CCI"] = (typical - sma_typ) / (0.015 * mad.replace(0, np.nan))

    # OBV
    direction = np.sign(out["Close"].diff().fillna(0))
    out["OBV"] = (direction * out["Volume"].fillna(0)).cumsum()

    return out

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import altair as alt
from pathlib import Path
from datetime import datetime
import os


from artifacts.loader import load_artifact
from artifacts.predictors import ensure_schema, build_features, predict_model


st.set_page_config(page_title="Stock Forecast (Hybrid)", page_icon="📈", layout="wide")
st.title("📈 Stock Forecast Web")


def _ensure_export_dir() -> Path:
    d = Path("artifacts/plots")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def save_altair_png(chart: alt.Chart, path: Path) -> tuple[bool, str]:
    """Save an Altair chart to a PNG file.
    Tries vl-convert first, then kaleido. Returns (ok, message)."""
    try:
        # Altair 5: prefer vl-convert if available
        chart.save(str(path), format="png", engine="vl-convert")
        return True, f"Saved with vl-convert → {path}"
    except Exception as e1:
        try:
            chart.save(str(path), format="png", engine="kaleido")
            return True, f"Saved with kaleido → {path}"
        except Exception as e2:
            msg = (
                f"Failed to export chart as PNG.\n"
                f"vl-convert error: {e1}\n"
                f"kaleido error: {e2}\n"
                f"Install 'vl-convert-python' (recommended) or 'kaleido'."
            )
            return False, msg


def resolve_artifact_path(preferred: str) -> str:
    p = Path(preferred)
    candidates = [
        p,
        Path("artifacts/notebooks/models") / p.name,
        Path("artifacts/notebooks") / p.name,
        Path("models") / p.name,
        Path("artifacts/models") / p.name,
    ]
    def _is_lfs_pointer(fp: Path) -> bool:
        try:
            if fp.is_file() and fp.stat().st_size < 2048:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(256)
                return head.startswith("version https://git-lfs.github.com/spec/v1")
        except Exception:
            pass
        return False

    for c in candidates:
        try:
            if c.exists():
                # Skip Git LFS pointer files; continue searching for real artifact
                if _is_lfs_pointer(c):
                    continue
                return str(c)
        except Exception:
            continue
    return str(p)


# Cache Yahoo Finance downloads (data cache)
@st.cache_data(show_spinner=False)
def yf_download_cached(ticker: str, period: str = "5y", interval: str = "1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)


# Cache artifact loading (resource cache) with mtime-based invalidation
@st.cache_resource(show_spinner=False)
def cached_load_artifact(resolved_path: str, mtime: float):
    # mtime participates in cache key; actual load uses path
    return load_artifact(resolved_path)


# UI: ticker via buttons and horizon
st.subheader("Select Ticker")
# Initialize session state once
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "BUMI"

ticker = st.selectbox(
    "Select ticker",
    ("BUMI", "ELSA", "DEWA"),
    index=("BUMI", "ELSA", "DEWA").index(st.session_state["ticker"]),
    key="ticker"
)
chosen_ticker = ticker
st.info(f"Active ticker: {ticker}")

# n_periods = st.slider("Forecast horizon (days)", 1, 5, 5)
n_periods = 5
debug_mode = False


# Download last 5 years daily data (cached)
raw = yf_download_cached(ticker + ".JK", period="5y", interval="1d")
raw = raw.reset_index()

if isinstance(raw.columns, pd.MultiIndex):
  raw.columns = [col[0] for col in raw.columns]

# Ensure a datetime column named 'ds'
if "Date" in raw.columns:
    raw = raw.rename(columns={"Date": "ds"})
else:
    # find any datetime-like column and rename it to ds
    for c in list(raw.columns):
        try:
            if pd.api.types.is_datetime64_any_dtype(raw[c]):
                raw = raw.rename(columns={c: "ds"})
                break
            # attempt parse sample values
            sample = raw[c].iloc[:5]
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().any():
                raw = raw.rename(columns={c: "ds"})
                break
        except Exception:
            continue

if "ds" not in raw.columns and isinstance(raw.index, pd.DatetimeIndex):
    raw["ds"] = raw.index
    raw = raw.reset_index(drop=True)

# Basic validation
min_history = 50
if len(raw) < min_history:
    st.error(f"Not enough history to build features (>= {min_history} rows required).")
    st.stop()

df = ensure_schema(raw)
feat = build_features(df)
if feat is None or feat.empty:
    st.error("Feature building failed; check data columns and history length.")
    st.stop()

# st.subheader("Latest data and features")
# st.dataframe(feat.tail(200))

# Predict button
st.markdown("---")
predict_button = st.button("🔮 Predict!", type="primary", use_container_width=True)

if predict_button:
    # Load artifacts and predict
    models = {
        "Hybrid"       : "models/" + chosen_ticker + "_hybrid.joblib",
        "Prophet"      : "models/" + chosen_ticker + "_prophet.joblib",
        "NHITS"        : "models/" + chosen_ticker + "_nhits.joblib",
        "NeuralProphet": "models/" + chosen_ticker + "_neuralprophet_meta.joblib",
    }

    results = []
    missing_artifacts = []
    nhits_scaler_used = False
    nhits_pred_df = None

    for name, path in models.items():
        resolved = resolve_artifact_path(path)
        if not Path(resolved).exists():
            missing_artifacts.append((name, path))
            continue
        try:
            # Use mtime to invalidate cache when artifact file changes
            try:
                mtime = Path(resolved).stat().st_mtime
            except Exception:
                mtime = -1.0
            artifact = cached_load_artifact(resolved, mtime)
            # Warn if scaler missing; tailor check per model type
            if debug_mode:
                mt = (artifact.get("model_type") or "").lower()
                missing = False
                if mt in {"prophet", "hybrid", "neuralprophet"}:
                    missing = not bool(artifact.get("scaler"))
                elif mt == "nhits":
                    # NHITS uses Darts Scaler objects for target and covariates
                    missing = not (bool(artifact.get("scaler_y")) and bool(artifact.get("scaler_cov")))
                if missing:
                    msg = (
                        f"{name}: artifact missing scaler — re-export including the fitted scaler to keep predictions on the correct price scale."
                        if mt in {"prophet", "hybrid", "neuralprophet"}
                        else f"{name}: artifact missing NHITS scalers (`scaler_y`/`scaler_cov`) — re-run NHITS notebook export to include them for consistent price-scale predictions."
                    )
                    st.warning(msg)
            pred = predict_model(artifact, feat, n_periods, debug=debug_mode)
            # Show a small debug note if NHITS used the persistence fallback
            if debug_mode and name == "NHITS":
                # Only show fallback message if it was actually used (check artifact flag)
                if artifact.get("_debug_nhits_fallback") and artifact["_debug_nhits_fallback"].get("reason"):
                    fb = artifact.get("_debug_nhits_fallback", {})
                    last_close = fb.get("last_close")
                    st.warning(f"⚠️ NHITS fallback active: using last Close={last_close} for horizon due to NaN forecast.")
                nhits_info = artifact.get("_debug_nhits_info")
                nhits_mismatch = artifact.get("_debug_nhits_mismatch")
                if nhits_info or nhits_mismatch:
                    with st.expander("NHITS debug details", expanded=False):
                        if nhits_info:
                            st.json(nhits_info)
                        if nhits_mismatch:
                            st.json({"feature_alignment": nhits_mismatch})
                # Capture scaler presence and NHITS forecast for later display/export
                nhits_scaler_used = bool(artifact.get("scaler_y"))
                nhits_pred_df = pred.copy()
            results.append(pred)
        except Exception as e:
            st.warning(f"{name} skipped: {e}")

    if missing_artifacts:
        st.warning("Artifacts missing: " + ", ".join([f"{n} → {p}" for n, p in missing_artifacts]))

    debug_mode = True
    # Output chart and table
    if results:
        forecast = pd.concat(results)
        hist = feat[["ds", "Close"]].set_index("ds")
        fc = forecast.pivot(index="ds", columns="model", values="yhat")

        # Rename columns to make it clear these are prices (IDR), not generic 'yhat'
        fc = fc.rename(columns={
            "Prophet": "Prophet (Price)",
            "Hybrid": "Hybrid (Price)",
            "NHITS": "NHITS (Price)",
            "NeuralProphet": "NeuralProphet (Price)",
        })
        hist = hist.rename(columns={"Close": "Actual (Price)"})

        # Heuristic rescaling fallback: if predictions are astronomically large compared to actual,
        # align by last value ratio to keep plot readable when scaler is missing.
        last_close = float(feat["Close"].iloc[-1])
        def _try_rescale(series):
            s = pd.to_numeric(series, errors="coerce")
            if s.isna().all():
                return series
            med_diff = (s - last_close).abs().median()
            if med_diff <= max(1e9, abs(last_close) * 1000):
                return series
            try:
                last_pred = float(s.dropna().iloc[-1])
            except Exception:
                return series
            if not np.isfinite(last_pred) or last_pred == 0:
                return series
            scale = last_pred / last_close
            if scale > 1e6 or scale < 1e-6:
                return series
            candidate = s / scale
            cand_med_diff = (candidate - last_close).abs().median()
            if cand_med_diff < max(1e6, abs(last_close) * 50):
                return candidate
            return series

        for col in list(fc.columns):
            fc[col] = _try_rescale(fc[col])

        chart_df = pd.concat([hist, fc], axis=1)

        st.subheader(f"Forecast charts for {ticker} over next {n_periods} days")

        # Zoom controls
        cols_ctrl = st.columns([1, 1, 2])
        # with cols_ctrl[0]:
        #     zoom_days = st.slider("Zoom window (days)", 30, 365, 180)
        # with cols_ctrl[1]:
        #     pad_pct = st.slider("Y-axis padding (%)", 0, 20, 5)
        pad_pct = 5
        zoom_days = 180
        pad = pad_pct / 100.0

        # Export controls
        export_dir = _ensure_export_dir()
        # export_enabled = st.checkbox("Enable chart export (PNG)", value=False)
        ts_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Compute recent window for y-domain and optional x-window
        hist_idx = chart_df.index
        try:
            cutoff = hist_idx.max() - pd.Timedelta(days=zoom_days)
        except Exception:
            cutoff = hist_idx.max()
        recent = chart_df[chart_df.index >= cutoff]

        # y-domain derived from recent values to "zoom into price"
        vals = recent.to_numpy().astype(float)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size:
            y_min = float(np.nanmin(finite_vals))
            y_max = float(np.nanmax(finite_vals))
            y_pad = (y_max - y_min) * pad if np.isfinite(y_max - y_min) else 0.0
            y_domain = [y_min - y_pad, y_max + y_pad]
        else:
            y_domain = None

        tabs = st.tabs(["Overview"] + list(fc.columns))

        # Overview tab: combined actual + all models
        with tabs[0]:
            st.write("Overview: Actual and all model predictions")
            # Build Altair overview chart with controlled y-axis domain
            ov_df = recent.reset_index().rename(columns={"ds": "Date"})
            ov_long = ov_df.melt(id_vars="Date", var_name="Series", value_name="Price")
            y_enc = alt.Y("Price:Q", title="Price", scale=alt.Scale(domain=y_domain) if y_domain else alt.Scale())
            x_enc = alt.X("Date:T", title="Date")
            overview_chart = (
                alt.Chart(ov_long)
                .mark_line()
                .encode(x=x_enc, y=y_enc, color=alt.Color("Series:N", title="Series"), tooltip=["Date:T", "Series:N", alt.Tooltip("Price:Q", format=",.2f")])
            ).interactive()
            st.altair_chart(overview_chart, use_container_width=True)
            st.subheader("Predicted prices (all models)")
            df_all = fc.reset_index().tail(n_periods)
            df_all.insert(0, "No", range(1, len(df_all) + 1))
            st.dataframe(df_all, hide_index=True)

        # Individual tabs per model: actual vs a single model
        for i, col in enumerate(fc.columns, start=1):
            with tabs[i]:
                st.write(f"Actual vs {col}")
                model_df = pd.concat([hist, fc[[col]]], axis=1)
                recent_model = model_df[model_df.index >= cutoff]
                m_vals = recent_model.to_numpy().astype(float)
                m_finite = m_vals[np.isfinite(m_vals)]
                if m_finite.size:
                    m_y_min = float(np.nanmin(m_finite))
                    m_y_max = float(np.nanmax(m_finite))
                    m_pad = (m_y_max - m_y_min) * pad if np.isfinite(m_y_max - m_y_min) else 0.0
                    m_domain = [m_y_min - m_pad, m_y_max + m_pad]
                else:
                    m_domain = None

                mdl_df = recent_model.reset_index().rename(columns={"ds": "Date"})
                mdl_long = mdl_df.melt(id_vars="Date", var_name="Series", value_name="Price")
                mdl_chart = (
                    alt.Chart(mdl_long)
                    .mark_line()
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Price:Q", title="Price", scale=alt.Scale(domain=m_domain) if m_domain else alt.Scale()),
                        color=alt.Color("Series:N", title="Series"),
                        tooltip=["Date:T", "Series:N", alt.Tooltip("Price:Q", format=",.2f")],
                    )
                ).interactive()
                st.altair_chart(mdl_chart, use_container_width=True)
                st.subheader("Predicted prices")
                df_single = fc[[col]].reset_index().tail(n_periods)
                df_single.insert(0, "No", range(1, len(df_single) + 1))
                st.dataframe(df_single, hide_index=True)

        st.subheader("Metrics")
        metrics_rows = []
        for name, path in models.items():
            resolved = resolve_artifact_path(path)
            try:
                art = joblib.load(resolved)
                metrics = art.get("metrics") if isinstance(art, dict) else None
                if metrics:
                    # Normalize metrics to scalar/string values for tabular display
                    safe_metrics = {}
                    try:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float, np.number)) or v is None:
                                safe_metrics[k] = v
                            elif isinstance(v, str):
                                safe_metrics[k] = v
                            else:
                                safe_metrics[k] = str(v)
                    except Exception:
                        # Fallback if metrics isn't a plain dict-like
                        safe_metrics = {"metrics": str(metrics)}
                    metrics_rows.append({"Model": name, **safe_metrics})
            except Exception:
                continue
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            # Insert a numbered column on the left starting from 1
            df_metrics.insert(0, "No", range(1, len(df_metrics) + 1))
            ordered_cols = ["No", "Model"] + [c for c in df_metrics.columns if c not in {"No", "Model"}]
            df_display = df_metrics[ordered_cols].set_index("No")
            st.dataframe(df_display)
        else:
            st.info("No metrics available to display as a table.")

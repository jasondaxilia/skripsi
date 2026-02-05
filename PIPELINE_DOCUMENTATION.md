# Dokumentasi Pipeline Machine Learning

## Aplikasi Stock Forecast Multi-Model

**Tanggal:** Februari 2026  
**Versi:** 1.0  
**Penulis:** [Nama Mahasiswa]

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Data Loading](#3-data-loading)
4. [Data Cleaning](#4-data-cleaning)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Data Augmentation](#6-data-augmentation)
7. [Data Splitting](#7-data-splitting)
8. [Model Definition](#8-model-definition)
9. [Model Training](#9-model-training)
10. [Model Evaluation](#10-model-evaluation)
11. [Forecasting / Inference](#11-forecasting--inference)
12. [Visualization](#12-visualization)
13. [User Interface](#13-user-interface)
14. [Model Comparison](#14-model-comparison)
15. [Kesimpulan](#15-kesimpulan)

---

## 1. Pendahuluan

Aplikasi ini merupakan sistem prediksi harga saham berbasis web yang menggunakan **lima model time series forecasting**:

| No  | Model                        | Tipe          | Karakteristik                                                   |
| --- | ---------------------------- | ------------- | --------------------------------------------------------------- |
| 1   | **Prophet**                  | Statistical   | Additive regression dengan trend, seasonality, holiday          |
| 2   | **Hybrid Prophet + XGBoost** | Ensemble      | Prophet untuk base forecast + XGBoost untuk residual correction |
| 3   | **N-HiTS**                   | Deep Learning | Neural Hierarchical Interpolation dengan multi-rate sampling    |
| 4   | **NeuralProphet**            | Deep Learning | Neural network implementation of Prophet                        |
| 5   | **N-BEATS**                  | Deep Learning | Neural Basis Expansion dengan residual stacking                 |

**Emiten yang didukung:** BUMI, ELSA, DEWA (Bursa Efek Indonesia)

---

## 2. Arsitektur Sistem

### 2.1 Diagram Alur Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (Offline - Notebooks)             │
├─────────────────────────────────────────────────────────────────────┤
│  Data Loading → Cleaning → Preprocessing → Splitting → Training    │
│       ↓             ↓            ↓             ↓           ↓       │
│   yfinance      ensure_      build_       TimeSeriesSplit  fit()   │
│                 schema()    features()                              │
│                                                                     │
│  Artifacts: {model, scaler, feature_columns, metrics} → .joblib    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE (Streamlit App)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ UI:      │   │ Data:    │   │ Prediction:  │   │ Output:     │  │
│  │ Ticker   │──▶│ Download │──▶│ load_artifact│──▶│ Visualize   │  │
│  │ Select   │   │ & Clean  │   │ predict_model│   │ & Compare   │  │
│  └──────────┘   └──────────┘   └──────────────┘   └─────────────┘  │
│                                                                     │
│  Models: Prophet │ Hybrid │ N-HiTS │ NeuralProphet │ N-BEATS       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Struktur File

```
skripsi/
├── app.py                          # Main Streamlit application
├── artifacts/
│   ├── __init__.py                 # Package exports
│   ├── features.py                 # Feature engineering
│   ├── loader.py                   # Artifact loading
│   ├── predictors.py               # Prediction logic per model
│   └── notebooks/                  # Training notebooks
│       ├── model_prophet.ipynb
│       ├── model_hybrid_prophet_xgboost.ipynb
│       ├── model_nhits.ipynb
│       ├── model_neuralprophet.ipynb
│       └── model_nbeats.ipynb
├── models/                         # Trained model artifacts
│   ├── {TICKER}_prophet.joblib
│   ├── {TICKER}_hybrid.joblib
│   ├── {TICKER}_nhits.joblib
│   ├── {TICKER}_nhits.darts
│   ├── {TICKER}_neuralprophet_meta.joblib
│   └── {TICKER}_nbeats.joblib
└── scripts/
    ├── validate_artifact.py        # Artifact validation
    └── test_nhits.py               # N-HiTS testing
```

---

## 3. Data Loading

### 3.1 Lokasi Code

| File     | Baris   | Fungsi                 |
| -------- | ------- | ---------------------- |
| `app.py` | 79-83   | `yf_download_cached()` |
| `app.py` | 108-112 | Pemanggilan download   |

### 3.2 Implementasi

```python
@st.cache_data(show_spinner=False)
def yf_download_cached(ticker: str, period: str = "5y", interval: str = "1d"):
    return yf.download(ticker, period=period, interval=interval,
                       progress=False, auto_adjust=True)

# Penggunaan
raw = yf_download_cached(ticker + ".JK", period="5y", interval="1d")
```

### 3.3 Penjelasan

- **Sumber data:** Yahoo Finance API via library `yfinance`
- **Periode:** 5 tahun data historis
- **Interval:** Harian (daily)
- **Caching:** Decorator `@st.cache_data` menyimpan hasil download untuk menghindari request berulang
- **Auto-adjust:** `auto_adjust=True` memastikan harga sudah disesuaikan dengan corporate action

### 3.4 Peran terhadap Performa Model

| Aspek           | Dampak                                                            |
| --------------- | ----------------------------------------------------------------- |
| Kualitas sumber | Yahoo Finance adalah sumber terpercaya dengan data yang konsisten |
| Volume data     | 5 tahun (~1250 trading days) cukup untuk menangkap pola seasonal  |
| Caching         | Meningkatkan responsivitas aplikasi tanpa mengorbankan freshness  |

---

## 4. Data Cleaning

### 4.1 Lokasi Code

| File                    | Baris   | Fungsi                                |
| ----------------------- | ------- | ------------------------------------- |
| `artifacts/features.py` | 8-56    | `ensure_schema()`                     |
| `app.py`                | 114-133 | Column renaming & MultiIndex handling |

### 4.2 Implementasi

```python
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has at least columns: ds (datetime), Close."""
    out = df.copy()

    # 1. Standarisasi nama kolom tanggal
    date_candidates = ["ds", "Date", "date", "Datetime", "datetime", "Timestamp"]
    found_col = None
    for col in date_candidates:
        if col in out.columns:
            found_col = col
            break

    # 2. Handle DatetimeIndex
    if found_col is None and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        found_col = "index"

    # 3. Rename to 'ds'
    if found_col and found_col != "ds":
        out = out.rename(columns={found_col: "ds"})

    # 4. Validasi kolom Close
    if "Close" not in out.columns:
        if "Adj Close" in out.columns:
            out = out.rename(columns={"Adj Close": "Close"})
        else:
            raise ValueError("Input data must contain a 'Close' price column.")

    # 5. Cleaning operations
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds"])           # Remove invalid dates
    out = out.sort_values("ds")               # Chronological order
    out = out.drop_duplicates(subset=["ds"])  # Remove duplicates

    return out
```

### 4.3 Penjelasan

| Operasi             | Tujuan                                                               |
| ------------------- | -------------------------------------------------------------------- |
| Standarisasi schema | Memastikan kolom tanggal bernama `ds` (format Prophet/NeuralProphet) |
| Handle MultiIndex   | Mengatasi output yfinance yang kadang berbentuk MultiIndex           |
| Dropna              | Menghapus baris dengan tanggal invalid                               |
| Drop duplicates     | Mencegah duplikasi data pada tanggal yang sama                       |
| Sort values         | Mengurutkan data secara kronologis                                   |

### 4.4 Peran terhadap Performa Model

- **Temporal ordering:** Data yang tidak terurut dapat menyebabkan model time series mempelajari pola yang salah
- **Konsistensi schema:** Memastikan kompatibilitas antar berbagai model
- **Missing value handling:** Mencegah error saat training dan inference

---

## 5. Data Preprocessing

### 5.1 Feature Engineering (Technical Indicators)

#### 5.1.1 Lokasi Code

| File                    | Baris  | Fungsi                                          |
| ----------------------- | ------ | ----------------------------------------------- |
| `artifacts/features.py` | 58-158 | `build_features()`                              |
| `artifacts/features.py` | 60-67  | Helper functions (`_ema()`, `_rsi()`, `_atr()`) |

#### 5.1.2 Daftar Fitur yang Dihasilkan

| Kategori           | Indikator       | Formula/Deskripsi                | Window      |
| ------------------ | --------------- | -------------------------------- | ----------- |
| **Lag Features**   | `Lag1` - `Lag5` | `Close.shift(k)`                 | 1-5 hari    |
| **Moving Average** | `MA20`          | Simple Moving Average            | 20 hari     |
| **Moving Average** | `MA50`          | Simple Moving Average            | 50 hari     |
| **Momentum**       | `MACD`          | EMA(12) - EMA(26)                | 12, 26 hari |
| **Momentum**       | `Signal Line`   | EMA(9) dari MACD                 | 9 hari      |
| **Momentum**       | `RSI`           | Relative Strength Index          | 14 hari     |
| **Volatility**     | `ATR`           | Average True Range               | 14 hari     |
| **Volatility**     | `BB_Upper`      | Bollinger Band Upper (MA20 + 2σ) | 20 hari     |
| **Volatility**     | `BB_Lower`      | Bollinger Band Lower (MA20 - 2σ) | 20 hari     |
| **Oscillator**     | `Stochastic_K`  | %K Stochastic                    | 14 hari     |
| **Oscillator**     | `Stochastic_D`  | %D (SMA of %K)                   | 3 hari      |
| **Trend**          | `CCI`           | Commodity Channel Index          | 20 hari     |
| **Volume**         | `OBV`           | On-Balance Volume                | Cumulative  |
| **Previous**       | `Prev Close`    | `Close.shift(1)`                 | 1 hari      |

#### 5.1.3 Implementasi Indikator Utama

```python
# RSI (Relative Strength Index)
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ATR (Average True Range)
def _atr(high, low, close, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

# Bollinger Bands
rolling_mean = close.rolling(20, min_periods=20).mean()
rolling_std = close.rolling(20, min_periods=20).std()
BB_Upper = rolling_mean + 2 * rolling_std
BB_Lower = rolling_mean - 2 * rolling_std
```

### 5.2 Feature Scaling

#### 5.2.1 Lokasi Code

| File                      | Baris   | Fungsi                                | Model           |
| ------------------------- | ------- | ------------------------------------- | --------------- |
| `artifacts/predictors.py` | 109-116 | Scaling di `_predict_prophet()`       | Prophet, Hybrid |
| `artifacts/predictors.py` | 160-166 | Scaling di `_predict_neuralprophet()` | NeuralProphet   |
| `artifacts/predictors.py` | 409-412 | Darts Scaler                          | N-HiTS          |
| `artifacts/predictors.py` | 646-649 | MinMaxScaler                          | N-BEATS         |

#### 5.2.2 Implementasi

```python
# Prophet/Hybrid/NeuralProphet: MinMaxScaler untuk features
if scaler is not None:
    Xf = future[feature_cols].values
    Xf_scaled = scaler.transform(Xf)
    future.loc[:, feature_cols] = Xf_scaled

# N-HiTS: Darts Scaler untuk target dan covariates
from darts.dataprocessing.transformers import Scaler as DartsScaler

if scaler_y is not None and isinstance(scaler_y, DartsScaler):
    ts_y_s = scaler_y.transform(ts_y)
if scaler_cov is not None and isinstance(scaler_cov, DartsScaler):
    ts_cov_s = scaler_cov.transform(ts_cov)

# N-BEATS: MinMaxScaler untuk univariate
if scaler is not None:
    close_scaled = scaler.transform(close_prices.reshape(-1, 1)).flatten()
```

#### 5.2.3 Perbedaan Scaling per Model

| Model         | Scaler Type  | Target Scaled?                     | Features Scaled? |
| ------------- | ------------ | ---------------------------------- | ---------------- |
| Prophet       | MinMaxScaler | ❌ No                              | ✅ Yes           |
| Hybrid        | MinMaxScaler | ❌ No                              | ✅ Yes           |
| N-HiTS        | Darts Scaler | ✅ Yes                             | ✅ Yes           |
| NeuralProphet | MinMaxScaler | ✅ Internal (`normalize='minmax'`) | ✅ Yes           |
| N-BEATS       | MinMaxScaler | ✅ Yes                             | N/A (univariate) |

### 5.3 Resampling & Frequency Handling

#### 5.3.1 Lokasi Code

| File                      | Baris   | Fungsi                  |
| ------------------------- | ------- | ----------------------- |
| `artifacts/predictors.py` | 54-61   | `_infer_freq_from_ds()` |
| `artifacts/predictors.py` | 361-362 | Resampling untuk N-HiTS |

#### 5.3.2 Implementasi

```python
def _infer_freq_from_ds(ds: pd.Series) -> str | None:
    """Infer frequency from date series."""
    try:
        freq = pd.infer_freq(pd.to_datetime(ds))
        if freq:
            return freq
    except Exception:
        pass
    return "B"  # Default: business daily

# N-HiTS: Ensure daily frequency with forward-fill
df_model_ready = df_model_ready.set_index("ds").asfreq("D").ffill().reset_index()
```

### 5.4 Windowing (untuk N-BEATS)

#### 5.4.1 Lokasi Code

| File                      | Baris   | Fungsi                                          |
| ------------------------- | ------- | ----------------------------------------------- |
| `artifacts/predictors.py` | 659-673 | Lookback window untuk autoregressive prediction |

#### 5.4.2 Implementasi

```python
lookback = config['lookback']  # e.g., 10 atau 20 hari
current_sequence = close_scaled[-lookback:].copy()

# Iterative multi-step forecasting
predictions_scaled = []
with torch.no_grad():
    for i in range(periods):
        x_input = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
        pred = model(x_input)
        pred_value = pred.cpu().numpy()[0, 0]

        predictions_scaled.append(pred_value)
        # Autoregressive: update sequence dengan prediksi
        current_sequence = np.append(current_sequence[1:], pred_value)
```

### 5.5 Peran Preprocessing terhadap Performa Model

| Aspek                    | Dampak pada Performa                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Technical Indicators** | Memberikan sinyal tambahan tentang momentum, volatilitas, dan tren yang meningkatkan akurasi prediksi  |
| **Scaling**              | Mencegah fitur dengan magnitude besar mendominasi pembelajaran; mempercepat konvergensi neural network |
| **Proper Frequency**     | Memastikan model tidak "bingung" dengan gap di data (weekend/holiday)                                  |
| **Windowing**            | Memungkinkan N-BEATS melakukan autoregressive forecasting secara iteratif                              |

---

## 6. Data Augmentation

### 6.1 Status

> **⚠️ DATA AUGMENTATION TIDAK DITERAPKAN DALAM APLIKASI INI**

Setelah menganalisis seluruh source code secara menyeluruh, tidak ditemukan implementasi data augmentation.

### 6.2 Alasan Tidak Diterapkan

#### 6.2.1 Karakteristik Data Time Series Finansial

Data harga saham bersifat **non-stationary** dan memiliki pola temporal yang unik. Teknik augmentasi yang umum digunakan di domain lain (seperti rotasi/flipping di computer vision) **tidak relevan** untuk data sekuensial.

#### 6.2.2 Preservasi Temporal Dependency

Model time series (Prophet, N-HiTS, N-BEATS) dirancang untuk **menangkap pola tren, seasonality, dan autokorelasi**. Augmentasi yang mengubah urutan atau magnitude dapat **merusak temporal dependency** yang kritis untuk forecasting.

#### 6.2.3 Volume Data Sudah Memadai

Data historis 5 tahun (~1250 trading days) sudah cukup untuk melatih model forecasting tanpa memerlukan augmentasi.

#### 6.2.4 Model Sudah Robust

- **Prophet/NeuralProphet:** Memiliki built-in handling untuk missing data dan outliers
- **N-HiTS:** Multi-rate sampling secara implisit menangkap variasi temporal
- **N-BEATS:** Residual learning membantu generalisasi

### 6.3 Contoh Augmentasi Opsional (Jika Diperlukan)

Berikut adalah beberapa teknik augmentasi yang **dapat ditambahkan** jika diperlukan:

```python
import numpy as np
from scipy.interpolate import CubicSpline

# 1. Jittering - Menambah noise gaussian kecil
def jitter(series: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add small gaussian noise to the series."""
    noise = np.random.normal(0, sigma * series.std(), len(series))
    return series + noise

# 2. Scaling - Mengubah magnitude secara proporsional
def scaling_augment(series: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Scale the series by a random factor."""
    scale = np.random.uniform(*scale_range)
    return series * scale

# 3. Window Slicing - Mengambil subsequence berbeda
def window_slicing(series: np.ndarray, slice_ratio: float = 0.9) -> np.ndarray:
    """Take a random subsequence of the series."""
    length = int(len(series) * slice_ratio)
    start = np.random.randint(0, len(series) - length)
    return series[start:start + length]

# 4. Magnitude Warping - Smooth random curve multiplication
def magnitude_warp(series: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """Warp magnitude with smooth random curve."""
    orig_steps = np.arange(len(series))
    random_warps = np.random.normal(1.0, sigma, knot + 2)
    warp_steps = np.linspace(0, len(series) - 1, knot + 2)
    warper = CubicSpline(warp_steps, random_warps)(orig_steps)
    return series * warper

# 5. Time Warping - Mengubah kecepatan temporal
def time_warp(series: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """Warp time axis with smooth random curve."""
    orig_steps = np.arange(len(series))
    random_warps = np.random.normal(1.0, sigma, knot + 2)
    warp_steps = np.linspace(0, len(series) - 1, knot + 2)
    time_warp = CubicSpline(warp_steps, random_warps)(orig_steps)
    time_warp = np.cumsum(time_warp)
    time_warp = (time_warp - time_warp.min()) / (time_warp.max() - time_warp.min()) * (len(series) - 1)
    return np.interp(orig_steps, time_warp, series)
```

### 6.4 Catatan Penting

> **⚠️ PERINGATAN:** Jika augmentasi diterapkan untuk data finansial, harus dilakukan dengan **sangat hati-hati** dan divalidasi secara menyeluruh. Augmentasi yang tidak tepat dapat menurunkan akurasi forecasting karena merusak pola temporal yang valid.

---

## 7. Data Splitting

### 7.1 Status

Data splitting **dilakukan di notebook training** (offline), bukan di aplikasi Streamlit. Aplikasi hanya melakukan inference menggunakan model yang sudah di-train.

### 7.2 Evidence dari Code

| File                           | Baris | Evidence                                                                         |
| ------------------------------ | ----- | -------------------------------------------------------------------------------- |
| `scripts/validate_artifact.py` | 59-64 | Fungsi `compute_metrics()` membandingkan `y_true` vs `y_pred`                    |
| Artifact structure             | -     | Setiap artifact menyimpan `metrics: {rmse, mae, mape, r2, directional_accuracy}` |

### 7.3 Pola Splitting untuk Time Series

```python
# PENTING: Time series HARUS menggunakan temporal split, BUKAN random split

# Opsi 1: Simple temporal split
train_size = int(len(df) * 0.8)
train = df[:train_size]    # Data lama untuk training
test = df[train_size:]     # Data baru untuk testing

# Opsi 2: Walk-forward validation (expanding window)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    # Train and evaluate...
```

### 7.4 Ilustrasi Temporal Split

```
Timeline: ──────────────────────────────────────────────────────────▶

Data:     [===== TRAIN (80%) =====][== TEST (20%) ==]
                                   ↑
                              Split point
                         (tidak boleh random!)

Walk-forward:
Fold 1:   [TRAIN     ][TEST]
Fold 2:   [TRAIN          ][TEST]
Fold 3:   [TRAIN               ][TEST]
Fold 4:   [TRAIN                    ][TEST]
Fold 5:   [TRAIN                         ][TEST]
```

### 7.5 Peran terhadap Performa Model

| Aspek              | Dampak                                                                       |
| ------------------ | ---------------------------------------------------------------------------- |
| **Temporal split** | Mencegah data leakage - model tidak boleh "melihat" masa depan saat training |
| **Walk-forward**   | Memberikan estimasi performa yang lebih robust di berbagai kondisi pasar     |
| **Stored metrics** | Memungkinkan evaluasi performa tanpa re-training                             |

---

## 8. Model Definition

### 8.1 Prophet

#### 8.1.1 Lokasi Code

| File                      | Baris   | Komponen                    |
| ------------------------- | ------- | --------------------------- |
| `artifacts/predictors.py` | 262-272 | Prediksi Prophet            |
| Artifact                  | -       | Key: `'prophet': <Prophet>` |

#### 8.1.2 Arsitektur

Prophet adalah **additive regression model** dengan komponen:

$$y(t) = g(t) + s(t) + h(t) + \sum_i \beta_i x_i(t) + \epsilon_t$$

| Komponen    | Simbol           | Deskripsi                                  |
| ----------- | ---------------- | ------------------------------------------ |
| Trend       | $g(t)$           | Piecewise linear atau logistic growth      |
| Seasonality | $s(t)$           | Fourier series untuk pola musiman          |
| Holiday     | $h(t)$           | Efek hari libur                            |
| Regressors  | $\beta_i x_i(t)$ | External regressors (technical indicators) |
| Error       | $\epsilon_t$     | Gaussian noise                             |

#### 8.1.3 Implementasi Prediksi

```python
def _predict_prophet(m, df, feature_cols, periods, scaler=None):
    future = _future_frame_from_last(df, periods)

    # Add and scale regressors if needed
    if feature_cols:
        for c in [c for c in feature_cols if c not in future.columns]:
            future[c] = df[c].iloc[-1]
        if scaler is not None:
            future[feature_cols] = scaler.transform(future[feature_cols])

    fc = m.predict(future)
    return fc[["ds", "yhat"]]
```

### 8.2 Hybrid Prophet + XGBoost

#### 8.2.1 Lokasi Code

| File                      | Baris   | Komponen                   |
| ------------------------- | ------- | -------------------------- |
| `artifacts/predictors.py` | 274-301 | Prediksi Hybrid            |
| Artifact                  | -       | Keys: `'prophet'`, `'xgb'` |

#### 8.2.2 Arsitektur

**Two-stage ensemble approach:**

```
Stage 1: Prophet Base Forecast
┌─────────────┐
│   Prophet   │ ──▶ base_forecast = trend + seasonality + regressors
└─────────────┘

Stage 2: XGBoost Residual Correction
┌─────────────┐
│   XGBoost   │ ──▶ residual = f(technical_indicators)
└─────────────┘

Final: yhat = base_forecast + residual
```

#### 8.2.3 Implementasi Prediksi

```python
if model_type == "hybrid":
    m = artifact.get("prophet")
    xgb = artifact.get("xgb")

    # Stage 1: Prophet base forecast
    base = _predict_prophet(m, df, feature_cols, periods, scaler=scaler)

    # Stage 2: XGBoost residual
    future = _future_frame_from_last(df, periods)
    Xf = future[feature_cols].values
    if scaler is not None:
        Xf = scaler.transform(Xf)
    residual = xgb.predict(Xf)

    # Combine
    out = base.copy()
    out["yhat"] = out["yhat"].values + residual
    return out
```

### 8.3 N-HiTS (Neural Hierarchical Interpolation for Time Series)

#### 8.3.1 Lokasi Code

| File                      | Baris   | Komponen                              |
| ------------------------- | ------- | ------------------------------------- |
| `artifacts/predictors.py` | 303-555 | Prediksi N-HiTS                       |
| Import                    | 9       | `from darts.models import NHiTSModel` |

#### 8.3.2 Arsitektur

N-HiTS menggunakan **multi-rate signal sampling** untuk menangkap pola di berbagai skala temporal:

```
Input Series ───▶ [Block 1: High Freq] ──┐
                  [Block 2: Med Freq]  ──┼──▶ Hierarchical Interpolation ──▶ Forecast
                  [Block 3: Low Freq]  ──┘

Each block has different expressiveness ratios and pooling kernel sizes
```

#### 8.3.3 Implementasi Prediksi

```python
if model_type == "nhits":
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler as DartsScaler

    # Load model
    m = NHiTSModel.load(artifact.get("nhits_path"))

    # Create TimeSeries objects
    ts_y = TimeSeries.from_dataframe(df_model_ready, "ds", "y", freq='D')
    ts_cov = TimeSeries.from_dataframe(cov_ext, "ds", feature_cols, freq='D')

    # Apply scalers
    ts_y_s = scaler_y.transform(ts_y)
    ts_cov_s = scaler_cov.transform(ts_cov)

    # Predict
    forecast_s = m.predict(n=periods, series=ts_y_s, past_covariates=ts_cov_s)

    # Inverse transform
    forecast = scaler_y.inverse_transform(forecast_s)
    return forecast
```

### 8.4 NeuralProphet

#### 8.4.1 Lokasi Code

| File                      | Baris   | Komponen                    |
| ------------------------- | ------- | --------------------------- |
| `artifacts/predictors.py` | 118-248 | `_predict_neuralprophet()`  |
| `artifacts/predictors.py` | 10-51   | `_np_expected_regressors()` |

#### 8.4.2 Arsitektur

NeuralProphet adalah **neural network implementation of Prophet** dengan penambahan:

- **AR-Net:** Autoregressive network untuk local patterns
- **Lagged regressors:** Support untuk fitur dengan lag
- **Learned embeddings:** Trend dan seasonality sebagai learned components

```
Input: ds, y, regressors
         │
         ▼
┌─────────────────────────────────┐
│  Trend (Piecewise Linear)      │
│  Seasonality (Fourier + NN)    │
│  AR-Net (Autoregressive)       │
│  Lagged Regressors             │
└─────────────────────────────────┘
         │
         ▼
Output: yhat1 (already denormalized)
```

#### 8.4.3 Implementasi Prediksi

```python
def _predict_neuralprophet(m, df, feature_cols, periods, scaler=None):
    df_in = df.copy()

    # Ensure 'y' column
    if "y" not in df_in.columns:
        df_in["y"] = df_in["Close"]

    # Scale features (not target - NeuralProphet handles internally)
    if scaler is not None:
        df_in[feature_cols] = scaler.transform(df_in[feature_cols])

    # Build future dataframe
    future = pd.DataFrame({"ds": future_dates, "y": np.nan})
    for c in feature_cols:
        future[c] = df_in[c].iloc[-1]  # Carry-forward

    # Predict - output is already in original scale
    fc = m.predict(future)
    return fc[["ds", "yhat1"]].rename(columns={"yhat1": "yhat"})
```

### 8.5 N-BEATS (Neural Basis Expansion Analysis for Time Series)

#### 8.5.1 Lokasi Code

| File                      | Baris   | Komponen                      |
| ------------------------- | ------- | ----------------------------- |
| `artifacts/predictors.py` | 557-696 | Custom PyTorch implementation |

#### 8.5.2 Arsitektur

N-BEATS menggunakan **doubly residual stacking** dengan basis expansion:

```
Input Sequence (lookback window)
         │
         ▼
┌─────────────────────────────────────────────┐
│  Stack 1                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Block 1 │─│ Block 2 │─│ Block 3 │       │
│  └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │             │
│  Backcast    Backcast    Backcast          │
│  Forecast    Forecast    Forecast          │
└─────────────────────────────────────────────┘
         │ (residual connection)
         ▼
┌─────────────────────────────────────────────┐
│  Stack 2, 3, ... N                          │
└─────────────────────────────────────────────┘
         │
         ▼
Sum of all Forecasts ──▶ Final Prediction
```

#### 8.5.3 Implementasi Model

```python
class NBeatsBlock(nn.Module):
    """Single N-BEATS Block with basis expansion"""
    def __init__(self, input_size, theta_size, basis_function, num_layers, layer_width):
        super().__init__()
        # Fully connected layers
        layers = [nn.Linear(input_size, layer_width), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(layer_width, layer_width), nn.ReLU()])
        self.fc = nn.Sequential(*layers)

        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(layer_width, theta_size)
        self.theta_f = nn.Linear(layer_width, theta_size)

    def forward(self, x):
        h = self.fc(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = self.basis_function(theta_b, self.input_size)
        forecast = self.basis_function(theta_f, self.input_size)
        return backcast, forecast


class NBEATSNet(nn.Module):
    """N-BEATS Network with multiple stacks"""
    def forward(self, x):
        forecast = torch.zeros(batch_size, self.output_size)
        residual = x

        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast  # Residual learning
                forecast = forecast + block_forecast

        return forecast
```

### 8.6 Perbandingan Model

| Aspek                | Prophet       | Hybrid        | N-HiTS           | NeuralProphet | N-BEATS          |
| -------------------- | ------------- | ------------- | ---------------- | ------------- | ---------------- |
| **Tipe**             | Statistical   | Ensemble      | Deep Learning    | Deep Learning | Deep Learning    |
| **Input**            | Multivariate  | Multivariate  | Multivariate     | Multivariate  | Univariate       |
| **Features**         | ✅ Regressors | ✅ Regressors | ✅ Covariates    | ✅ Regressors | ❌ Close only    |
| **Interpretable**    | ✅ High       | ✅ Medium     | ❌ Low           | ✅ Medium     | ❌ Low           |
| **Handling Missing** | ✅ Built-in   | ✅ Built-in   | ⚠️ Requires fill | ✅ Built-in   | ⚠️ Requires fill |
| **Training Speed**   | Fast          | Fast          | Slow             | Medium        | Slow             |

---

## 9. Model Training

### 9.1 Status

Training **dilakukan offline** di Jupyter Notebooks, bukan di aplikasi Streamlit. Model yang sudah di-train disimpan sebagai artifact (`.joblib`).

### 9.2 Lokasi Notebooks

| Model         | Notebook                                                 |
| ------------- | -------------------------------------------------------- |
| Prophet       | `artifacts/notebooks/model_prophet.ipynb`                |
| Hybrid        | `artifacts/notebooks/model_hybrid_prophet_xgboost.ipynb` |
| N-HiTS        | `artifacts/notebooks/model_nhits.ipynb`                  |
| NeuralProphet | `artifacts/notebooks/model_neuralprophet.ipynb`          |
| N-BEATS       | `artifacts/notebooks/model_nbeats.ipynb`                 |

### 9.3 Artifact Loading

#### 9.3.1 Lokasi Code

| File                  | Baris  | Fungsi                   |
| --------------------- | ------ | ------------------------ |
| `artifacts/loader.py` | 28-130 | `load_artifact()`        |
| `app.py`              | 85-88  | `cached_load_artifact()` |

#### 9.3.2 Implementasi

```python
def load_artifact(path: str) -> Dict[str, Any]:
    """Load a saved artifact with multiple fallback strategies."""
    p = Path(path)

    # Try joblib first
    try:
        obj = joblib.load(str(p))
    except Exception:
        # Fallback: memory-mapped mode
        try:
            obj = joblib.load(str(p), mmap_mode="r")
        except Exception:
            # Fallback: standard pickle
            with open(p, "rb") as f:
                obj = pickle.load(f)

    # Special handling for NeuralProphet
    if obj.get("model_type") == "neuralprophet":
        if "model_dir" in obj and "neuralprophet" not in obj:
            from neuralprophet import NeuralProphet
            obj["neuralprophet"] = NeuralProphet.load(obj["model_dir"])

    return obj
```

### 9.4 Struktur Artifact

```python
# Prophet Artifact
{
    'model_type': 'prophet',
    'prophet': <Prophet object>,
    'scaler': <MinMaxScaler>,
    'feature_columns': ['MA20', 'MA50', 'RSI', ...],
    'metrics': {'rmse': 0.05, 'mae': 0.03, ...}
}

# Hybrid Artifact
{
    'model_type': 'hybrid',
    'prophet': <Prophet object>,
    'xgb': <XGBRegressor object>,
    'scaler': <MinMaxScaler>,
    'feature_columns': [...],
    'metrics': {...}
}

# N-HiTS Artifact
{
    'model_type': 'nhits',
    'nhits_path': 'models/BUMI_nhits.darts',
    'scaler_y': <DartsScaler>,
    'scaler_cov': <DartsScaler>,
    'feature_columns': [...],
    'metrics': {...}
}

# N-BEATS Artifact
{
    'model_type': 'nbeats',
    'config': {'input_size': 10, 'output_size': 1, ...},
    'model_state_dict': <OrderedDict>,
    'scaler': <MinMaxScaler>,
    'metrics': {...}
}

# NeuralProphet Artifact
{
    'model_type': 'neuralprophet',
    'neuralprophet': <NeuralProphet object>,  # atau 'model_dir'
    'scaler': <MinMaxScaler>,
    'feature_columns': [...],
    'metrics': {...}
}
```

---

## 10. Model Evaluation

### 10.1 Lokasi Code

| File                           | Baris   | Fungsi                                        |
| ------------------------------ | ------- | --------------------------------------------- |
| `scripts/validate_artifact.py` | 38-65   | `compute_metrics()`, `directional_accuracy()` |
| `app.py`                       | 359-393 | Display metrics dari artifact                 |

### 10.2 Metrik yang Digunakan

#### 10.2.1 Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interpretasi:** Rata-rata magnitude error dalam unit asli (IDR). Semakin kecil semakin baik.

#### 10.2.2 Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interpretasi:** Seperti MAE tetapi lebih sensitif terhadap outlier. Berguna untuk risk assessment.

#### 10.2.3 Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Interpretasi:** Percentage error yang memudahkan perbandingan antar skala harga berbeda.

#### 10.2.4 R-squared (R²)

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

**Interpretasi:** Proporsi variance yang dijelaskan oleh model. Range 0-1, semakin tinggi semakin baik.

#### 10.2.5 Directional Accuracy (DA)

$$\text{DA} = \frac{1}{n-1} \sum_{i=2}^{n} \mathbb{1}[\text{sign}(\Delta y_i) = \text{sign}(\Delta \hat{y}_i)]$$

**Interpretasi:** Proporsi prediksi yang benar dalam memprediksi arah pergerakan (naik/turun). **Sangat penting untuk trading strategy.**

### 10.3 Implementasi

```python
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy - correct prediction of up/down movement."""
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    if len(dy_true) == 0 or len(dy_pred) == 0:
        return np.nan
    return float(np.mean(np.sign(dy_true) == np.sign(dy_pred)))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) if np.all(y_true != 0) else np.nan
    r2 = r2_score(y_true, y_pred)
    dir_acc = directional_accuracy(y_true.values, y_pred.values)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "DirAcc": dir_acc
    }
```

### 10.4 Peran terhadap Performa Model

| Metrik   | Use Case                                                                      |
| -------- | ----------------------------------------------------------------------------- |
| **MAE**  | Evaluasi umum, mudah diinterpretasi                                           |
| **RMSE** | Ketika error besar harus dihindari (risk-sensitive)                           |
| **MAPE** | Perbandingan antar saham dengan harga berbeda                                 |
| **R²**   | Seberapa baik model menjelaskan variasi data                                  |
| **DA**   | **Paling penting untuk trading** - lebih baik benar arah daripada nilai eksak |

---

## 11. Forecasting / Inference

### 11.1 Lokasi Code

| File                      | Baris   | Fungsi                          |
| ------------------------- | ------- | ------------------------------- |
| `artifacts/predictors.py` | 255-748 | `predict_model()` - Main router |
| `artifacts/predictors.py` | 64-91   | `_future_frame_from_last()`     |
| `app.py`                  | 170-207 | Prediction loop                 |

### 11.2 Future Frame Generation

```python
def _future_frame_from_last(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Create future dataframe by extending dates and forward-filling features."""
    last_date = pd.to_datetime(df["ds"].iloc[-1])
    freq = _infer_freq_from_ds(df["ds"]) or "D"

    # Generate future dates
    future_dates = pd.date_range(
        last_date + pd.tseries.frequencies.to_offset(freq),
        periods=periods,
        freq=freq
    )

    future = pd.DataFrame({"ds": future_dates})

    # Carry-forward features (last known values)
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            future[c] = df[c].iloc[-1]

    return future
```

### 11.3 Prediction Router

```python
def predict_model(artifact: Dict, df: pd.DataFrame, periods: int, debug: bool = False):
    """Route prediction to appropriate model handler."""
    model_type = artifact.get("model_type", "unknown")

    if model_type == "prophet":
        return _predict_prophet(...)

    if model_type == "hybrid":
        base = _predict_prophet(...)
        residual = xgb.predict(...)
        return base + residual

    if model_type == "nhits":
        forecast = m.predict(n=periods, series=ts_y_s, past_covariates=ts_cov_s)
        return scaler_y.inverse_transform(forecast)

    if model_type == "neuralprophet":
        return _predict_neuralprophet(...)

    if model_type == "nbeats":
        # Autoregressive multi-step
        for i in range(periods):
            pred = model(x_input)
            current_sequence = update_sequence(pred)
        return inverse_transform(predictions)

    raise ValueError(f"Unsupported model_type: {model_type}")
```

### 11.4 Fallback Mechanism

Jika prediksi menghasilkan NaN atau error, sistem memiliki **fallback mechanism**:

```python
# N-HiTS fallback: persistence (last close value)
if not has_finite_predictions:
    last_close = df["Close"].iloc[-1]
    fallback_df = pd.DataFrame({
        "ds": future_dates,
        "y": [last_close] * periods
    })
    artifact["_debug_nhits_fallback"] = {
        "reason": "All-NaN forecast; using persistence fallback",
        "last_close": last_close,
    }
```

### 11.5 Peran terhadap Performa

| Aspek                      | Dampak                                             |
| -------------------------- | -------------------------------------------------- |
| **Inverse scaling**        | Memastikan output dalam skala harga original (IDR) |
| **Business day frequency** | Menghasilkan tanggal yang realistis (skip weekend) |
| **Fallback mechanism**     | Mencegah aplikasi crash jika prediksi gagal        |

---

## 12. Visualization

### 12.1 Lokasi Code

| File     | Baris   | Fungsi              |
| -------- | ------- | ------------------- |
| `app.py` | 273-357 | Altair charts       |
| `app.py` | 28-45   | `save_altair_png()` |

### 12.2 Implementasi Chart

```python
# Overview chart: semua model + actual
ov_long = ov_df.melt(id_vars="Date", var_name="Series", value_name="Price")

overview_chart = (
    alt.Chart(ov_long)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Price:Q", title="Price",
                scale=alt.Scale(domain=y_domain)),
        color=alt.Color("Series:N", title="Series"),
        tooltip=[
            "Date:T",
            "Series:N",
            alt.Tooltip("Price:Q", format=",.2f")
        ]
    )
).interactive()

st.altair_chart(overview_chart, use_container_width=True)
```

### 12.3 Fitur Visualisasi

| Fitur                   | Deskripsi                         | Code Reference           |
| ----------------------- | --------------------------------- | ------------------------ |
| **Y-axis auto-zoom**    | Domain dihitung dari data terbaru | Lines 280-290            |
| **Interactive tooltip** | Hover untuk melihat nilai         | `.encode(tooltip=[...])` |
| **Tabs**                | Overview + individual model       | `st.tabs([...])`         |
| **Price formatting**    | Format ribuan dengan koma         | `format=",.2f"`          |
| **Zoom window**         | Configurable (default 180 hari)   | `zoom_days = 180`        |

### 12.4 Peran terhadap Performa Model

- **Visualisasi** membantu **interpretasi** dan **validasi visual** prediksi
- Memudahkan deteksi **anomali** atau prediksi yang tidak masuk akal
- Mendukung **model comparison** secara visual

---

## 13. User Interface

### 13.1 Lokasi Code

| File     | Baris   | Widget              |
| -------- | ------- | ------------------- |
| `app.py` | 15-16   | Page config & title |
| `app.py` | 92-103  | Ticker selectbox    |
| `app.py` | 151-152 | Predict button      |
| `app.py` | 299     | Tabs                |

### 13.2 Komponen UI

```python
# Page configuration
st.set_page_config(
    page_title="Stock Forecast (Hybrid)",
    page_icon="📈",
    layout="wide"
)
st.title("📈 Stock Forecast Web")

# Ticker selection
ticker = st.selectbox(
    "Select ticker",
    ("BUMI", "ELSA", "DEWA"),
    key="ticker"
)
st.info(f"Active ticker: {ticker}")

# Predict button
predict_button = st.button(
    "🔮 Predict!",
    type="primary",
    use_container_width=True
)

# Result tabs
tabs = st.tabs(["Overview"] + list(fc.columns))

# Data tables
st.dataframe(df_all, hide_index=True)

# Notifications
st.warning("...")  # Warning messages
st.error("...")    # Error messages
st.expander("...")  # Collapsible debug info
```

### 13.3 Daftar UI Components

| Component          | Fungsi                   | State Management             |
| ------------------ | ------------------------ | ---------------------------- |
| `st.selectbox`     | Pemilihan ticker saham   | `st.session_state["ticker"]` |
| `st.button`        | Trigger prediksi         | -                            |
| `st.tabs`          | Navigasi antar model     | -                            |
| `st.dataframe`     | Tabel prediksi           | -                            |
| `st.altair_chart`  | Visualisasi chart        | -                            |
| `st.warning/error` | Notifikasi               | -                            |
| `st.expander`      | Debug info (collapsible) | -                            |
| `st.info`          | Informational messages   | -                            |

---

## 14. Model Comparison

### 14.1 Lokasi Code

| File                           | Baris   | Fungsi                            |
| ------------------------------ | ------- | --------------------------------- |
| `app.py`                       | 359-393 | Metrics table                     |
| `app.py`                       | 233-242 | Pivot forecast untuk perbandingan |
| `scripts/validate_artifact.py` | -       | Offline validation script         |

### 14.2 Implementasi

```python
# Pivot untuk perbandingan side-by-side
forecast = pd.concat(results)
fc = forecast.pivot(index="ds", columns="model", values="yhat")

# Rename columns untuk clarity
fc = fc.rename(columns={
    "Prophet": "Prophet (Price)",
    "Hybrid": "Hybrid (Price)",
    "NHITS": "NHITS (Price)",
    "NeuralProphet": "NeuralProphet (Price)",
    "N-BEATS": "N-BEATS (Price)",
})

# Metrics table dari artifacts
metrics_rows = []
for name, path in models.items():
    art = joblib.load(resolved)
    metrics = art.get("metrics")
    if metrics:
        metrics_rows.append({"Model": name, **metrics})

df_metrics = pd.DataFrame(metrics_rows)
st.dataframe(df_metrics)
```

### 14.3 Fitur Comparison

| Fitur                      | Deskripsi                                       |
| -------------------------- | ----------------------------------------------- |
| **Side-by-side chart**     | Semua model di satu grafik dengan warna berbeda |
| **Metrics table**          | RMSE, MAE, MAPE, R², DA per model               |
| **Individual tabs**        | Zoom per model vs actual price                  |
| **Predicted prices table** | Nilai prediksi untuk n_periods hari ke depan    |

### 14.4 Peran terhadap Performa

- Memungkinkan **ensemble selection** berdasarkan metrik
- User dapat memilih model terbaik sesuai use case:
  - Prioritas **RMSE rendah** untuk risk management
  - Prioritas **Directional Accuracy tinggi** untuk trading signals

---

## 15. Kesimpulan

### 15.1 Ringkasan Pipeline

| Tahap             | Status             | File/Lokasi                                                    |
| ----------------- | ------------------ | -------------------------------------------------------------- |
| Data Loading      | ✅ Implemented     | `app.py` - yfinance                                            |
| Data Cleaning     | ✅ Implemented     | `features.py` - `ensure_schema()`                              |
| Preprocessing     | ✅ Implemented     | `features.py` - `build_features()`, scaling di `predictors.py` |
| Data Augmentation | ❌ Not Implemented | N/A (tidak diperlukan untuk time series)                       |
| Data Splitting    | ✅ Implemented     | Offline di notebooks                                           |
| Model Definition  | ✅ Implemented     | 5 models di `predictors.py`                                    |
| Model Training    | ✅ Implemented     | Offline di notebooks                                           |
| Model Evaluation  | ✅ Implemented     | 5 metrik (MAE, RMSE, MAPE, R², DA)                             |
| Forecasting       | ✅ Implemented     | `predictors.py` - `predict_model()`                            |
| Visualization     | ✅ Implemented     | Altair charts di `app.py`                                      |
| User Interface    | ✅ Implemented     | Streamlit widgets di `app.py`                                  |
| Model Comparison  | ✅ Implemented     | Tabs + metrics table                                           |

### 15.2 Kelebihan Arsitektur

1. **Modular Design:** Pemisahan jelas antara feature engineering, loading, dan prediction
2. **Multi-Model Support:** 5 model berbeda untuk perbandingan komprehensif
3. **Caching:** Efisien dengan `@st.cache_data` dan `@st.cache_resource`
4. **Fallback Mechanism:** Robust terhadap prediction failures
5. **Interpretable Metrics:** Directional Accuracy untuk trading relevance

### 15.3 Rekomendasi Pengembangan

1. **Hyperparameter Tuning:** Implementasi grid search/Optuna di notebooks
2. **Ensemble Method:** Combine predictions dari multiple models
3. **Real-time Updates:** Scheduled data refresh
4. **Confidence Intervals:** Prediction intervals untuk uncertainty quantification
5. **Backtesting Framework:** Walk-forward validation dengan profit/loss simulation

---

**Dokumen ini dibuat untuk keperluan akademis dan dapat digunakan sebagai referensi dalam sidang skripsi.**

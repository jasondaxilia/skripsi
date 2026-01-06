# Panduan Export Artifact Model

Dokumen ini menjelaskan struktur **artifact joblib** yang konsisten untuk setiap tipe model dalam proyek skripsi.

## 📋 Struktur Umum Artifact

Setiap artifact yang di-export harus memiliki struktur dictionary dengan key-key berikut:

```python
{
    'model_type': str,           # Tipe model: 'prophet', 'hybrid', 'nhits', 'nbeats', 'neuralprophet'
    'model': object,             # Model instance (atau key spesifik seperti 'prophet', 'nhits_path')
    'scaler': object,            # MinMaxScaler atau Darts Scaler (None jika tidak pakai scaling)
    'feature_columns': list,     # List nama kolom fitur yang digunakan
    'metrics': dict,             # Dictionary berisi metrik evaluasi
}
```

---

## 🔵 1. Model Prophet

**File:** `{EMITEN}_prophet.joblib`

```python
artifact = {
    'model_type': 'prophet',
    'prophet': model_prophet,        # Prophet model instance
    'scaler': scaler,                # MinMaxScaler instance (fitted)
    'feature_columns': regressors,   # List feature names
    'metrics': {
        'rmse': float(rmse_prophet),
        'mae': float(mae_prophet),
        'mape': float(mape_prophet),
        'r2': float(r2_prophet),
        'directional_accuracy': float(da_prophet),
    },
}
```

**Catatan:**
- ✅ Menggunakan `MinMaxScaler` untuk normalisasi fitur teknikal
- ✅ Target (`y`/`Close`) **TIDAK** dinormalisasi
- ✅ `feature_columns` berisi list nama fitur yang digunakan sebagai regressors

---

## 🟢 2. Model Hybrid (Prophet + XGBoost)

**File:** `{EMITEN}_hybrid.joblib`

```python
artifact = {
    'model_type': 'hybrid',
    'prophet': model_prophet,        # Prophet model instance
    'xgb': xgb_model,                # XGBoost model instance
    'scaler': scaler,                # MinMaxScaler instance (fitted)
    'feature_columns': feature_cols, # List feature names
    'metrics': {
        'rmse': float(rmse_hybrid),
        'mae': float(mae_hybrid),
        'mape': float(mape_hybrid),
        'r2': float(r2_hybrid),
        'directional_accuracy': float(da_hybrid),
    },
}
```

**Catatan:**
- ✅ Menggunakan **dua model**: Prophet (base) + XGBoost (residual correction)
- ✅ Scaler yang sama digunakan untuk kedua model
- ✅ `feature_columns` sama dengan yang digunakan di Prophet

---

## 🟣 3. Model N-HiTS (Darts)

**File:** `{EMITEN}_nhits.joblib` + `{EMITEN}_nhits.darts`

```python
# Simpan model Darts secara terpisah
nhits_path = export_dir / f"{EMITEN}_nhits.darts"
model.save(str(nhits_path))

# Artifact joblib
artifact = {
    'model_type': 'nhits',
    'nhits_path': f"models/{EMITEN}_nhits.darts",  # Relative path ke .darts file
    'scaler_y': scaler_y,           # Darts Scaler untuk target
    'scaler_cov': scaler_cov,       # Darts Scaler untuk covariates
    'feature_columns': selected_features,  # List feature names
    'ticker': ticker,               # Symbol saham (e.g., 'BUMI.JK')
    'metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(da),
    },
}
```

**Catatan:**
- ✅ Model N-HiTS disimpan sebagai **file .darts terpisah** menggunakan `model.save()`
- ✅ Artifact joblib hanya menyimpan **path** ke file .darts
- ✅ Menggunakan **Darts Scaler** (bukan sklearn MinMaxScaler)
- ✅ Ada dua scaler: `scaler_y` (target) dan `scaler_cov` (covariates)

---

## 🟠 4. Model N-BEATS (Darts)

**File:** `{EMITEN}_nbeats.joblib`

```python
artifact = {
    'model_type': 'nbeats',
    'model': nbeats_model,          # NBEATSModel instance
    'scaler': None,                 # N-BEATS uses raw prices (no scaling)
    'feature_columns': [],          # Empty list (univariate model)
    'metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(da),
    },
    'model_params': {               # Optional: simpan hyperparameters
        'input_chunk_length': 10,
        'output_chunk_length': 1,
        'generic_architecture': True,
        'num_stacks': 5,
        'num_blocks': 1,
        'num_layers': 2,
        'layer_widths': 64,
    }
}
```

**Catatan:**
- ✅ N-BEATS adalah **univariate model** (hanya menggunakan Close price)
- ✅ **Tidak menggunakan scaler** (raw prices)
- ✅ `feature_columns` adalah **list kosong** `[]`
- ✅ Optional: simpan `model_params` untuk dokumentasi

---

## 🔧 Template Export Code

### Template untuk Prophet / Hybrid

```python
from pathlib import Path
import joblib

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / 'requirements.txt').exists() or (p / 'app.py').exists() or (p / '.git').exists():
            return p
    return start

repo_root = find_repo_root(Path.cwd())
export_dir = (repo_root / 'models')
export_dir.mkdir(parents=True, exist_ok=True)

artifact = {
    'model_type': 'prophet',  # atau 'hybrid'
    'prophet': model_prophet,
    'scaler': scaler,
    'feature_columns': regressors,
    'metrics': {
        'rmse': float(rmse_prophet),
        'mae': float(mae_prophet),
        'mape': float(mape_prophet),
        'r2': float(r2_prophet),
        'directional_accuracy': float(da_prophet),
    },
}

artifact_path = export_dir / f'{emiten}_prophet.joblib'
joblib.dump(artifact, str(artifact_path))
print(f'✅ Saved artifact: {artifact_path.resolve()}')
```

### Template untuk N-HiTS

```python
from pathlib import Path
import joblib
from darts.models import NHiTSModel

repo_root = find_repo_root(Path.cwd())
export_dir = (repo_root / 'models')
export_dir.mkdir(parents=True, exist_ok=True)

# Simpan model .darts
nhits_path = export_dir / f"{emiten}_nhits.darts"
model.save(str(nhits_path))

# Simpan artifact joblib
artifact = {
    'model_type': 'nhits',
    'nhits_path': f"models/{emiten}_nhits.darts",
    'scaler_y': scaler_y,
    'scaler_cov': scaler_cov,
    'feature_columns': selected_features,
    'ticker': ticker,
    'metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(da),
    },
}

artifact_path = export_dir / f"{emiten}_nhits.joblib"
joblib.dump(artifact, str(artifact_path))
print(f'✅ Saved N-HiTS artifact: {artifact_path.resolve()}')
print(f'✅ Saved N-HiTS model: {nhits_path.resolve()}')
```

### Template untuk N-BEATS

```python
from pathlib import Path
import joblib
from darts.models import NBEATSModel

repo_root = find_repo_root(Path.cwd())
export_dir = (repo_root / 'models')
export_dir.mkdir(parents=True, exist_ok=True)

artifact = {
    'model_type': 'nbeats',
    'model': nbeats_model,
    'scaler': None,
    'feature_columns': [],
    'metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(da),
    },
    'model_params': {
        'input_chunk_length': 10,
        'output_chunk_length': 1,
        'generic_architecture': True,
        'num_stacks': 5,
        'num_blocks': 1,
        'num_layers': 2,
        'layer_widths': 64,
    }
}

artifact_path = export_dir / f'{emiten}_nbeats.joblib'
joblib.dump(artifact, str(artifact_path))
print(f'✅ Saved N-BEATS artifact: {artifact_path.resolve()}')
```

---

## ✅ Checklist Export

Sebelum menjalankan export, pastikan:

- [ ] Variabel `emiten` sudah didefinisikan (e.g., `'BUMI'`, `'DEWA'`, `'ELSA'`)
- [ ] Model sudah dilatih (`model_prophet`, `xgb_model`, `model`, dll.)
- [ ] Scaler sudah di-fit pada training data (`scaler`, `scaler_y`, `scaler_cov`)
- [ ] Feature columns tersimpan dalam variabel (`regressors`, `feature_cols`, `selected_features`)
- [ ] Metrics sudah dihitung (`rmse`, `mae`, `r2`, `mape`, `da`)
- [ ] Directory `models/` dibuat dengan `mkdir(parents=True, exist_ok=True)`

---

## 📁 Lokasi File Output

Semua artifact disimpan di:

```
{repo_root}/models/
├── BUMI_prophet.joblib
├── BUMI_hybrid.joblib
├── BUMI_nhits.joblib
├── BUMI_nhits.darts         # File terpisah untuk N-HiTS
├── BUMI_nhits.darts.ckpt    # Checkpoint PyTorch Lightning
├── BUMI_nbeats.joblib
├── DEWA_prophet.joblib
├── DEWA_hybrid.joblib
... dst
```

---

## 🔍 Verifikasi Artifact

Untuk memverifikasi artifact yang sudah disimpan:

```python
import joblib

# Load artifact
artifact = joblib.load('models/BUMI_prophet.joblib')

# Cek struktur
print("Model type:", artifact['model_type'])
print("Has scaler:", artifact['scaler'] is not None)
print("Features:", len(artifact['feature_columns']))
print("Metrics:", artifact['metrics'])
```

---

**Last updated:** January 2026

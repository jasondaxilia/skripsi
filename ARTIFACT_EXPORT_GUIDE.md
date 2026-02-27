# Panduan Export Artifact Model

Dokumen ini mendefinisikan struktur standar untuk export artifact model (`.joblib`) yang digunakan dalam proyek ini.

## Struktur Umum

Setiap artifact direpresentasikan sebagai dictionary dengan key wajib berikut:

```python
{
    'model_type': str,           # 'prophet', 'hybrid', 'nhits', 'nbeats', 'neuralprophet'
    'model': object,             # Instance model atau key spesifik ('prophet', 'nhits_path')
    'scaler': object,            # Object scaler (atau None)
    'feature_columns': list,     # Nama fitur yang digunakan
    'metrics': dict,             # Metrik evaluasi model
}
```

---

## Definisi Artifact per Model

### 1. Prophet (`{EMITEN}_prophet.joblib`)
```python
artifact = {
    'model_type': 'prophet',
    'prophet': model_prophet,
    'scaler': scaler,
    'feature_columns': regressors,
    'metrics': {
        'rmse': float(rmse), 'mae': float(mae), 'mape': float(mape),
        'r2': float(r2), 'directional_accuracy': float(da)
    },
}
```
- Menggunakan `MinMaxScaler` untuk fitur teknikal.
- Target tidak dinormalisasi.

### 2. Hybrid Prophet + XGBoost (`{EMITEN}_hybrid.joblib`)
```python
artifact = {
    'model_type': 'hybrid',
    'prophet': model_prophet,
    'xgb': xgb_model,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'metrics': { ... },
}
```
- Prophet bertindak sebagai base model, XGBoost untuk residual correction.
- Scaler dan fitur dibagikan untuk kedua model.

### 3. N-HiTS (`{EMITEN}_nhits.joblib` & `.darts`)
```python
# Simpan model Darts (.darts)
model.save(str(export_dir / f"{EMITEN}_nhits.darts"))

artifact = {
    'model_type': 'nhits',
    'nhits_path': f"models/{EMITEN}_nhits.darts",
    'scaler_y': scaler_y,
    'scaler_cov': scaler_cov,
    'feature_columns': selected_features,
    'ticker': ticker,
    'metrics': { ... },
}
```
- Model utama disimpan sebagai file `.darts`. Artifact `.joblib` hanya menyimpan _path_ ke model tersebut.
- Menggunakan Darts Scaler secara terpisah untuk target dan covariates.

### 4. N-BEATS (`{EMITEN}_nbeats.joblib`)
```python
artifact = {
    'model_type': 'nbeats',
    'model': nbeats_model,
    'scaler': None,
    'feature_columns': [],
    'metrics': { ... },
    'model_params': { ... } # Opsional
}
```
- Model univariate menggunakan harga mentah (tanpa scaler).

---

## Skrip Export & Verifikasi

Semua artifact harus disimpan di dalam folder `models/` pada root repository.

**Contoh penyimpanan:**
```python
import joblib
from pathlib import Path

export_dir = Path.cwd() / 'models'
export_dir.mkdir(parents=True, exist_ok=True)

artifact_path = export_dir / f"{emiten}_{model_type}.joblib"
joblib.dump(artifact, str(artifact_path))
```

**Contoh load & verifikasi:**
```python
import joblib

artifact = joblib.load('models/BUMI_prophet.joblib')
print(f"Type: {artifact['model_type']} | Features: {len(artifact['feature_columns'])}")
```

# Dokumentasi Pipeline Machine Learning
## Sistem Peramalan Harga Saham Multi-Model

**Tanggal:** Februari 2026
**Versi:** 1.0

---

## 1. Pendahuluan
Dokumen ini menguraikan arsitektur prapemrosesan, pelatihan, dan inferensi untuk aplikasi peramalan harga saham berbasis web. Sistem ini menggunakan lima model deret waktu untuk memprediksi harga saham pada tiga emiten (BUMI, ELSA, DEWA).

Model yang diimplementasikan meliputi:
1. **Prophet**: Model regresi aditif dengan komponen tren dan musiman.
2. **Hybrid Prophet + XGBoost**: Pendekatan ansambel (Prophet sebagai *base forecast*, XGBoost sebagai koreksi residu).
3. **N-HiTS**: Pembelajaran mendalam dengan arsitektur interpolasi hierarkis.
4. **NeuralProphet**: Implementasi *neural network* dari algoritma Prophet.
5. **N-BEATS**: Arsitektur *neural basis expansion* dengan koneksi residu ganda.

## 2. Arsitektur Sistem
Sistem terbagi menjadi dua fase utama:
1. **Fase Pelatihan (Luring/Offline)**: Dilakukan melalui Jupyter Notebook (`artifacts/notebooks/`). Melibatkan ekstraksi data, prapemrosesan, pembagian data, pelatihan model, dan penyimpanan luaran model (*artifact*).
2. **Fase Inferensi (Daring/Online)**: Aplikasi web interaktif berbasis Streamlit (`app.py`). Sistem memuat *artifact* model yang telah dilatih untuk melakukan prediksi dan visualisasi.

## 3. Alur Pemrosesan Data

### 3.1 Akuisisi dan Pembersihan Data
Data historis harga saham diunduh melalui API Yahoo Finance dengan periode 5 tahun. Proses pembersihan meliputi:
- Standarisasi format penamaan kolom parameter tanggal menjadi `ds` dan kolom target menjadi `Close`.
- Penghapusan baris data kosong (NaN) dan duplikasi waktu.
- Pengaturan urutan data secara kronologis.

### 3.2 Rekayasa Fitur (*Feature Engineering*)
Selain mengandalkan harga tutupan (*Close*), model (kecuali N-BEATS) memanfaatkan sejumlah indikator teknikal sebagai fitur tambahan:
- *Lag Features* (1-5 hari)
- *Moving Averages* (SMA 20, 50)
- *Momentum* (MACD, RSI, Stochastic)
- *Volatility* (ATR, Bollinger Bands)
- *Volume* (OBV)

### 3.3 Transformasi dan Skala (*Scaling*)
Normalisasi dilakukan berbeda tergantung karakteristik model:
- **Prophet & Hybrid Prophet+XGBoost**: Menggunakan `MinMaxScaler` dari scikit-learn untuk fitur teknikal. Data target tidak dinormalisasi.
- **N-HiTS**: Menggunakan `DartsScaler` secara terpisah untuk data target dan fitur kovariat.
- **NeuralProphet**: Fitur teknikal disekalakan (`MinMaxScaler`), sementara data target dinormalisasi secara internal.
- **N-BEATS**: Model univariat murni yang hanya menggunakan data harga mentah tanpa fitur tambahan, dinormalisasi dengan jendela (*windowing*).

### 3.4 Pembagian Data (*Data Splitting*)
Pemisahan data pelatihan dan pengujian dilakukan secara temporal (berdasarkan waktu kronologis) di fase luring. Hal ini dilakukan untuk mencegah kebocoran data (*data leakage*) yang umum terjadi jika pemisahan dilakukan secara acak pada analisis deret waktu.

## 4. Definisi Model

### 4.1 Prophet
Model statistik yang mendekomposisi data deret waktu ke dalam parameter tren, musiman, dan liburan. Fitur teknikal ditambahkan sebagai *regressor* linier independen. Prediksi akhir dihasilkan melalui modul `Predictor` yang memanfaatkan koefisien tersebut.

### 4.2 Hybrid Prophet + XGBoost
Model ini memisahkan prediksi ke dalam dua proses. Tahap pertama mendapatkan prediksi dasar melalui Prophet. Sisa selisih (residu) antara prediksi dasar dan nilai aktual selanjutnya dimodelkan menggunakan XGBoost yang bertindak sebagai korektor nonlinear dari indikator teknikal. Prediksi final merupakan penjumlahan dari prediksi Prophet dan residu XGBoost.

### 4.3 N-HiTS
Arsitektur *deep learning* yang mengandalkan teknik *sampling* sinyal pada berbagai kecepatan (*multi-rate signal sampling*). Model memproses informasi menggunakan pemisahan temporal yang berbeda melalui modul-modul frekuensi tinggi, sedang, dan rendah, sebelum merangkum hasil prediksi dalam proses interpolasi.

### 4.4 NeuralProphet
Memiliki arsitektur yang terinspirasi dari Prophet namun menggunakan jaringan saraf tiruan (AR-Net). NeuralProphet memungkinkan pemodelan autoregresif lokal dan penyertaan parameter lag yang lebih kompleks, tanpa mengorbankan interpretabilitas modul tren dan musiman.

### 4.5 N-BEATS
N-BEATS menggunakan *windowing process*, yaitu mekanisme di mana masa depan diprediksi melalui blok residu ganda yang mengamati parameter harga $n$ hari ke belakang (*lookback window*). Proses *autoregressive multi-step forecasting* menjamin model ini secara bertahap memperbarui untai data dengan prediksinya sendiri.

## 5. Implementasi *Artifact* dan Inferensi
Agar performa tinggi aplikasi Streamlit terjaga tanpa menjalankan ulang pelatihan, model dikemas dalam *artifact* berformat `.joblib` berserta daftar parameter pendukung lainnya (seperti *scaler*, kolom rekayasa fitur, dan rekam metrik performanya). Khusus N-HiTS, bobot parameter jaringan saraf disimpan ke dalam ekstensi spesifik `.darts`. 

Sub-modul `loader.py` menangani proses deserialisasi dinamis melalui *caching mechanism*, sedangkan sub-modul `predictors.py` bertindak sebagai *router* inferensi per model.

## 6. Evaluasi Performa
Ketepatan prediksi diukur menggunakan metrik statistik standar pada data uji saat fase pelatihan luring:
- **MAE** (*Mean Absolute Error*)
- **RMSE** (*Root Mean Squared Error*)
- **MAPE** (*Mean Absolute Percentage Error*)
- **R²** (*R-Squared*)
- **DA** (*Directional Accuracy*): Mengukur persentase ketepatan model dalam menentukan arah tren (naik/turun), suatu parameter krusial untuk keputusan perdagangan.

Seluruh rekam metrik evaluasi direkapitulasi dan divisualisasikan berdampingan dalam panel *Model Comparison* di aplikasi agar memfasilitasi pengguna untuk menetapkan keputusan teknikal yang beralasan.

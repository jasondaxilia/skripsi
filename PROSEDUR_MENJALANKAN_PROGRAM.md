# Buku Panduan Pengguna (User Manual)

Dokumen panduan instalasi dan cara menjalankan aplikasi peramalan saham berbasis Jupyter Notebook dan Streamlit.

## 1. Persiapan Sistem
Pastikan komputer yang digunakan memiliki:
1. Python versi 3.10 atau 3.11 yang sudah terpasang.
2. Koneksi internet aktif untuk mengunduh pustaka (library) dan menarik data saham dari Yahoo Finance.

## 2. Langkah Instalasi

1. Buka Command Prompt (Windows) atau Terminal (Mac/Linux).
2. Arahkan *directory* ke dalam folder proyek skripsi ini, contohnya:
   ```cmd
   cd C:/Users/Nama/skripsi
   ```
3. (Opsional tapi disarankan) Buat *virtual environment* agar pustaka yang di-instal tidak bentrok dengan proyek Python lain:
   ```cmd
   python -m venv env
   env\Scripts\activate
   ```
4. Instal semua library yang dibutuhkan dengan perintah:
   ```cmd
   pip install -r requirements.txt
   ```

## 3. Menjalankan Aplikasi Web (Streamlit)

Aplikasi utama untuk melihat hasil prediksi berjalan di peramban web (*browser*). 

1. Pastikan Anda masih berada di dalam folder proyek dan *virtual environment* sudah aktif.
2. Jalankan perintah berikut:
   ```cmd
   streamlit run app.py
   ```
3. Browser akan otomatis terbuka dan menampilkan halaman aplikasi (biasanya di alamat `http://localhost:8501`).

**Cara Menggunakan Aplikasi:**
- Pada halaman utama, pilih kode saham (BUMI, ELSA, atau DEWA) dari menu *dropdown*.
- Klik tombol **Predict!**.
- Tunggu beberapa saat. Aplikasi akan menampilkan grafik harga aktual berserta garis prediksi 5 hari ke depan dari keseluruhan model.
- Anda bisa melihat perbandingan performa akurasi (seperti tabel metrik dasar RMSE, MAE, R²) di bagian bawah halaman.

## 4. Melatih Ulang Model (Luring / Offline)

Jika Anda ingin memperbarui model dengan data saham yang paling baru, Anda harus melatih ulang model-model tersebut.

1. Jalankan perintah berikut di terminal:
   ```cmd
   python scripts/run_all_models.py
   ```
2. Tunggu prosesnya selesai. Proses ini akan menjalankan kelima algoritma (Prophet, Hybrid, N-HiTS, NeuralProphet, N-BEATS) untuk ketiga saham.
3. Lama waktu bergantung pada kecepatan komputer (bisa memakan waktu 10-30 menit).
4. Hasil pelatihan akan otomatis disimpan ke dalam folder `models/` menggantikan file yang lama.

## 5. Validasi Hasil Pelatihan

(Opsional) Untuk sekadar memastikan bahwa semua model telah berhasil dilatih ulang dan angkanya normal, Anda dapat menjalankan skrip pengecekan:

```cmd
python scripts/validate_artifact.py
```
Perintah ini akan mencetak skor metrik (R², RMSE, dll) ke layar terminal tanpa perlu membuka aplikasi web.

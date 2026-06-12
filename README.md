# 🌋 Sistem Informasi Spasio-Temporal Kebencanaan Gempa Bumi Indonesia (2016–2026)

Aplikasi dashboard interaktif berbasis web ini dirancang menggunakan kerangka kerja **Streamlit** untuk mengeksplorasi, memetakan, dan menganalisis data aktivitas seismik regional di wilayah teritorial Indonesia dalam rentang waktu makro 10 tahun. 

Sistem ini mengintegrasikan repositori data langsung dari **API U.S. Geological Survey (USGS)** menggunakan metode *Monthly Data Chunking* dan diperkaya dengan algoritma *Machine Learning* **Isolation Forest** untuk mendeteksi anomali/pencilan spasio-temporal secara *real-time*.

---

## 📑 Fitur Utama Aplikasi

1. **Dashboard Spasio-Temporal (Halaman 1):**
   * **Kontrol Temporal Dinamis:** Filter interaktif berbasis *Chained Filter* menggunakan Dropdown Tahun (2016–2026) yang terhubung langsung secara dinamis dengan Slider Komponen Bulan.
   * **Visualisasi Spasial Interaktif:** Peta episentrum gempa bumi menggunakan Mapbox (`carto-positron`) dengan ukuran marker proporsional terhadap nilai magnitudo.
   * **Analisis Dampak Sosial (Saran Dosen):** Integrasi data pembersihan dampak korban teridentifikasi di lapangan yang disajikan lewat *Metric Card*, peta *tooltip*, dan grafik donat distribusi tingkat keparahan bencana (`🟢 Tanpa Korban`, `🟡 Dampak Ringan`, `🔴 Dampak Berat`).

2. **Deteksi Anomali Machine Learning (Halaman 2):**
   * **On-the-Fly Modeling:** Pelatihan model *unsupervised learning* menggunakan algoritma **Isolation Forest** langsung di dalam *runtime* aplikasi.
   * **Slider Kontaminasi Dinamis:** Memungkinkan pengguna/penguji mengubah rasio sensitivitas pencilan secara langsung untuk mengisolasi gempa bumi ekstrem.
   * **Peta Kontras & Matriks Informasi:** Memetakan klaster gempa normal vs anomali secara geografis dan menyajikannya dalam tabel data interaktif yang diurutkan berdasarkan tingkat keekstremannya (*anomaly score*).

---

## 🛠️ Persyaratan Sistem & Dependensi

Proyek ini dikembangkan menggunakan bahasa pemrograman **Python 3.8+**. Seluruh dependensi pustaka yang dibutuhkan telah dicatat di dalam berkas `requirements.txt`, meliputi:
* `streamlit>=1.35.0` (Framework Antarmuka Multi-Page)
* `pandas>=2.0.0` (Prapemrosesan & Manipulasi Data Deret Waktu)
* `plotly>=5.15.0` (Mesin Grafik & Peta Spasial Interaktif)
* `scikit-learn>=1.3.0` (Penyedia Algoritma Isolation Forest)
* `numpy>=1.24.0` (Komputasi Matriks Numerik & Simulasi)

---

## 🚀 Panduan Instalasi dan Eksekusi Lokal

Ikuti instruksi di bawah ini untuk menjalankan aplikasi di perangkat lokal Anda:

### 1. Kloning atau Siapkan Folder Proyek
Pastikan struktur berkas di dalam folder proyek Anda telah tersusun seperti berikut:
```text
📂 proyek-gempa-indonesia/
├── 📄 app.py                           # Berkas utama aplikasi Streamlit (Multi-Page)
├── 📄 dataset_gempa_indonesia_10_tahun.csv  # Dataset makro hasil akuisisi bulanan
├── 📄 requirements.txt                 # Daftar dependensi library Python
└── 📄 README.md                        # Dokumentasi teknis sistem (Berkas ini)

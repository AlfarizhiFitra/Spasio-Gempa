import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest

# ==============================================================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Spasio-Temporal Gempa Indonesia",
    page_icon="🌋",
    layout="wide"
)

# ==============================================================================
# 2. FUNGSI MEMBACA DATASET UTAMA (Menggunakan Caching)
# ==============================================================================
@st.cache_data
def load_data():
    # Membaca dataset 10 tahun hasil akuisisi cicilan bulanan
    df = pd.read_csv("dataset_gempa_indonesia_10_tahun.csv")
    
    # Memastikan format data waktu valid
    df['waktu_kejadian'] = pd.to_datetime(df['waktu_kejadian'])
    df['tahun_periode'] = df['waktu_kejadian'].dt.strftime('%Y')
    df['bulan_nama'] = df['waktu_kejadian'].dt.strftime('%B') # Mengambil nama bulan (e.g., January, February)
    df['bulan_angka'] = df['waktu_kejadian'].dt.month # Untuk kebutuhan sorting bulan agar berurutan
    
    return df

# Memuat data ke aplikasi
try:
    df_gempa = load_data()
except FileNotFoundError:
    st.error("File 'dataset_gempa_indonesia_10_tahun.csv' tidak ditemukan. Silakan jalankan skrip akuisisi data 10 tahun terlebih dahulu!")
    st.stop()


# ==============================================================================
# HALAMAN 1: DASHBOARD VISUALISASI SPASIO-TEMPORAL
# ==============================================================================
def halaman_dashboard():
    st.title("🗺️ Dashboard Eksplorasi Spasio-Temporal Gempa Indonesia")
    st.markdown("Halaman ini berfungsi sebagai media eksplorasi data visual makro dari aktivitas seismik teritorial Indonesia periode 2016–2026.")
    st.markdown("---")

    # Filter di Sidebar
    st.sidebar.subheader("⚙️ Filter Eksplorasi Data")
    
    # Filter 1: Dropdown untuk Memilih Tahun
    list_tahun = sorted(df_gempa['tahun_periode'].unique(), reverse=True)
    tahun_terpilih = st.sidebar.selectbox("Pilih Tahun Pengamatan:", options=list_tahun, index=0)
    
    # Menyaring data internal untuk mencari bulan apa saja yang tersedia di tahun terpilih tersebut
    df_tahun_ini = df_gempa[df_gempa['tahun_periode'] == tahun_terpilih].sort_values(by='bulan_angka')
    list_bulan = df_tahun_ini['bulan_nama'].unique()
    
    # Filter 2: Slider untuk Memilih Bulan (Dinamis mengikuti tahun yang dipilih)
    if len(list_bulan) > 0:
        bulan_terpilih = st.sidebar.select_slider(
            "Geser untuk Pilih Bulan:", 
            options=list_bulan,
            value=list_bulan[-1] # Default ke bulan terakhir yang tersedia
        )
    else:
        st.sidebar.warning("Tidak ada data bulan tersedia untuk tahun ini.")
        bulan_terpilih = None

    # Filter 3: Slider Minimal Magnitudo
    min_mag = float(df_gempa['magnitudo'].min())
    max_mag = float(df_gempa['magnitudo'].max())
    magnitudo_terpilih = st.sidebar.slider(
        "Minimal Magnitudo (SR):", 
        min_value=min_mag, 
        max_value=max_mag, 
        value=4.5, 
        step=0.1
    )

    # Proses Penyaringan Data Akhir secara Dinamis
    df_filtered = df_gempa[
        (df_gempa['tahun_periode'] == tahun_terpilih) & 
        (df_gempa['bulan_nama'] == bulan_terpilih) & 
        (df_gempa['magnitudo'] >= magnitudo_terpilih)
    ]

    # Ringkasan Metrik Utama (Terintegrasi dengan Data Pembersihan Korban Jiwa)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Kejadian Gempa", len(df_filtered))
    with col2:
        total_korban = df_filtered['korban_teridentifikasi'].sum() if not df_filtered.empty else 0
        st.metric("Total Korban Teridentifikasi", f"{total_korban} Jiwa", delta="Saran Dosen", delta_color="off")
    with col3:
        gempa_terbesar = df_filtered['magnitudo'].max() if not df_filtered.empty else 0
        st.metric("Magnitudo Tertinggi", f"{gempa_terbesar} SR")
    with col4:
        rata_kedalaman = round(df_filtered['kedalaman_km'].mean(), 1) if not df_filtered.empty else 0
        st.metric("Rata-rata Kedalaman", f"{rata_kedalaman} Km")

    st.markdown("---")

    # Layout Dua Kolom untuk Visualisasi
    kolom_peta, kolom_grafik = st.columns([2, 1])

    with kolom_peta:
        st.subheader(f"📍 Episentrum Spasial: {bulan_terpilih} {tahun_terpilih}")
        if not df_filtered.empty:
            fig_map = px.scatter_mapbox(
                df_filtered, 
                lat="latitude", 
                lon="longitude", 
                size="magnitudo", 
                color="tingkat_dampak",      
                color_discrete_map={"🟢 Tanpa Korban": "#10b981", "🟡 Dampak Ringan": "#f59e0b", "🔴 Dampak Berat": "#ef4444"},
                hover_name="lokasi_deskripsi",
                hover_data={"waktu_kejadian": True, "magnitudo": True, "kedalaman_km": True, "korban_teridentifikasi": True},
                zoom=3.5, 
                center={"lat": -2.5, "lon": 118.0}, 
                mapbox_style="carto-positron", 
                height=500
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Tidak ada aktivitas gempa yang memenuhi kriteria filter bulan ini.")

    with kolom_grafik:
        st.subheader("📊 Distribusi Tingkat Dampak")
        if not df_filtered.empty:
            # Grafik donat untuk melihat proporsi tingkat dampak korban gempa
            fig_pie = px.pie(
                df_filtered, 
                names='tingkat_dampak',
                color='tingkat_dampak',
                color_discrete_map={"🟢 Tanpa Korban": "#10b981", "🟡 Dampak Ringan": "#f59e0b", "🔴 Dampak Berat": "#ef4444"},
                hole=0.4,
                height=450
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Grafik tidak tersedia.")


# ==============================================================================
# HALAMAN 2: DETEKSI ANOMALI SPASIO-TEMPORAL (MACHINE LEARNING)
# ==============================================================================
def halaman_ml_anomali():
    st.title("🧠 Pengayaan Machine Learning: Deteksi Anomali Spasio-Temporal")
    st.markdown(
        "Halaman ini mengimplementasikan algoritma **Isolation Forest** untuk mendeteksi pencilan (*outlier*) "
        "berdasarkan variabel spasial (koordinat) dan fisis (magnitudo & kedalaman) dari seluruh data historis 10 tahun."
    )
    st.markdown("---")

    # 1. Parameter Kontaminasi Model di Sidebar
    st.sidebar.subheader("⚙️ Parameter Model ML")
    kontaminasi = st.sidebar.slider(
        "Rasio Kontaminasi Ekspektasi (Anomali %):",
        min_value=0.01, 
        max_value=0.15, 
        value=0.05, 
        step=0.01,
        help="Estimasi persentase data ekstrem yang dianggap sebagai anomali dari keseluruhan dataset makro."
    )

    # 2. Proses Pemodelan Alur Machine Learning secara Real-Time
    with st.spinner("Model Machine Learning sedang menganalisis tren data makro..."):
        # Ekstraksi matriks fitur numerik (X)
        X = df_gempa[['latitude', 'longitude', 'magnitudo', 'kedalaman_km']]
        
        # Inisialisasi dan fitting objek model Isolation Forest
        clf = IsolationForest(contamination=kontaminasi, random_state=42)
        df_gempa['status_anomali'] = clf.fit_predict(X)
        df_gempa['skor_anomali'] = clf.score_samples(X)
        
        # Transformasi label hasil prediksi untuk kebutuhan visual
        df_gempa['label_anomali'] = df_gempa['status_anomali'].apply(
            lambda x: '🚨 Anomali/Ekstrem' if x == -1 else '🟢 Normal'
        )

    # 3. Output Ringkasan Temuan Deteksi Model
    total_anomali = len(df_gempa[df_gempa['status_anomali'] == -1])
    total_normal = len(df_gempa) - total_anomali
    persen_anomali = int(kontaminasi * 100)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Total Gempa Terdeteksi Anomali", 
            value=f"{total_anomali} Kejadian", 
            delta=f"{persen_anomali}% dari total data 10 tahun", 
            delta_color="inverse"
        )
    with col2:
        st.metric(
            label="Total Gempa Berkategori Normal", 
            value=f"{total_normal} Kejadian"
        )

    st.markdown("---")

    # 4. Peta Hasil Klasifikasi Model ML
    st.subheader("🗺️ Peta Distribusi Geografis Hasil Deteksi Anomali")
    
    fig_ml_map = px.scatter_mapbox(
        df_gempa, 
        lat="latitude", 
        lon="longitude", 
        size="magnitudo",
        color="label_anomali", 
        color_discrete_map={
            '🚨 Anomali/Ekstrem': '#ef4444', # Merah menyala untuk anomali
            '🟢 Normal': '#10b981'          # Hijau tenang untuk normal
        },
        hover_name="lokasi_deskripsi",
        hover_data={
            "waktu_kejadian": True, 
            "magnitudo": True, 
            "kedalaman_km": True, 
            "korban_teridentifikasi": True, # Menyertakan data korban di tooltip peta ML
            "skor_anomali": ":.3f"          # Membatasi desimal skor anomali agar rapi
        },
        zoom=3.8, 
        center={"lat": -2.5, "lon": 118.0}, 
        mapbox_style="carto-positron", 
        height=550
    )
    st.plotly_chart(fig_ml_map, use_container_width=True)

    st.markdown("---")

    # 5. Penyajian Matriks Deteksi dalam Bentuk Tabel Data Informasi (Sorted by Extremity)
    st.subheader("📋 Tabel Sampel Kejadian Gempa Kategori Anomali")
    st.markdown("Daftar di bawah menampilkan data gempa yang paling cepat diisolasi oleh algoritma berdasarkan tingkat keekstremannya:")
    
    df_hanya_anomali = df_gempa[df_gempa['status_anomali'] == -1][
        ['waktu_kejadian', 'lokasi_deskripsi', 'magnitudo', 'kedalaman_km', 'korban_teridentifikasi', 'skor_anomali']
    ].sort_values(by='skor_anomali', ascending=True) # Urutkan dari yang skornya paling rendah (paling anomali)
    
    # Render tabel interaktif Streamlit
    st.dataframe(df_hanya_anomali, use_container_width=True)

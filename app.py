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
    # Membaca data mentah hasil akuisisi
    df = pd.read_csv("dataset_gempa_indonesia_2023_2026.csv")
    
    # Memastikan format kolom penanggalan valid
    df['waktu_kejadian'] = pd.to_datetime(df['waktu_kejadian'])
    
    # Mengamankan fitur ekstrak waktu jika belum ada di file CSV mentah
    df['tahun_periode'] = df['waktu_kejadian'].dt.strftime('%Y')
    df['bulan_periode'] = df['waktu_kejadian'].dt.strftime('%Y-%m')
    
    return df

# Memuat data awal ke dalam aplikasi
try:
    df_gempa = load_data()
except FileNotFoundError:
    st.error("File 'dataset_gempa_indonesia_2023_2026.csv' tidak ditemukan. Silakan jalankan skrip akuisisi data terlebih dahulu!")
    st.stop()


# ==============================================================================
# HALAMAN 1: DASHBOARD VISUALISASI SPASIO-TEMPORAL
# ==============================================================================
def halaman_dashboard():
    st.title("🗺️ Dashboard Eksplorasi Spasio-Temporal Gempa Indonesia")
    st.markdown("Halaman ini berfungsi sebagai media eksplorasi data visual dari aktivitas seismik teritorial Indonesia.")
    st.markdown("---")

    # Filter di Sidebar khusus untuk halaman dashboard
    st.sidebar.subheader("⚙️ Filter Eksplorasi")
    
    tahun_list = sorted(df_gempa['tahun_periode'].unique())
    tahun_terpilih = st.sidebar.select_slider(
        "Pilih Tahun Pengamatan:", 
        options=tahun_list, 
        value=tahun_list[-1]
    )
    
    min_mag = float(df_gempa['magnitudo'].min())
    max_mag = float(df_gempa['magnitudo'].max())
    magnitudo_terpilih = st.sidebar.slider(
        "Minimal Magnitudo (SR):", 
        min_value=min_mag, 
        max_value=max_mag, 
        value=4.0, 
        step=0.1
    )

    # Proses Filter Data secara Dinamis berdasarkan Input Pengguna
    df_filtered = df_gempa[
        (df_gempa['tahun_periode'] == tahun_terpilih) & 
        (df_gempa['magnitudo'] >= magnitudo_terpilih)
    ]

    # Ringkasan Metrik Perubahan Data Dinamis
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Kejadian Gempa Terfilter", len(df_filtered))
    with col2:
        gempa_terbesar = df_filtered['magnitudo'].max() if not df_filtered.empty else 0
        st.metric("Magnitudo Tertinggi", f"{gempa_terbesar} SR")
    with col3:
        rata_kedalaman = round(df_filtered['kedalaman_km'].mean(), 1) if not df_filtered.empty else 0
        st.metric("Rata-rata Kedalaman", f"{rata_kedalaman} Km")

    st.markdown("---")

    # Peta Sebaran Kerapatan Spasial
    st.subheader("📍 Peta Sebaran Episentrum Spasial")
    if not df_filtered.empty:
        fig_map = px.scatter_mapbox(
            df_filtered, 
            lat="latitude", 
            lon="longitude", 
            size="magnitudo", 
            color="kedalaman_km",      
            color_continuous_scale="inferno", 
            hover_name="lokasi_deskripsi",
            hover_data={"waktu_kejadian": True, "magnitudo": True, "kedalaman_km": True},
            zoom=3.8, 
            center={"lat": -2.5, "lon": 118.0}, 
            mapbox_style="carto-positron", 
            height=500
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Tidak ada aktivitas gempa yang memenuhi kriteria filter.")

    st.markdown("---")

    # Tren Distribusi Analisis Temporal Bulanan
    st.subheader("📈 Tren Temporal Frekuensi Gempa Per Bulan")
    if not df_filtered.empty:
        tren_bulanan = df_filtered.groupby('bulan_periode').size().reset_index(name='Jumlah Kejadian')
        fig_line = px.line(
            tren_bulanan, 
            x='bulan_periode', 
            y='Jumlah Kejadian',
            labels={'bulan_periode': 'Bulan Pengamatan', 'Jumlah Kejadian': 'Frekuensi Gempa'},
            markers=True, 
            color_discrete_sequence=['#ff4b4b'], 
            height=300
        )
        st.plotly_chart(fig_line, use_container_width=True)


# ==============================================================================
# HALAMAN 2: DETEKSI ANOMALI SPASIO-TEMPORAL (MACHINE LEARNING)
# ==============================================================================
def halaman_ml_anomali():
    st.title("🧠 Pengayaan Machine Learning: Deteksi Anomali Spasio-Temporal")
    st.markdown(r"Halaman ini mengimplementasikan algoritma **Isolation Forest** untuk mendeteksi aktivitas gempa ekstrem/pencilan (*outlier*) berdasarkan koordinat spasial, magnitudo, dan kedalaman.")
    st.markdown("---")

    # Parameter Kontaminasi Model di Sidebar
    st.sidebar.subheader("⚙️ Parameter Model ML")
    kontaminasi = st.sidebar.slider(
        "Rasio Kontaminasi Ekspektasi (Anomali %):",
        min_value=0.01, 
        max_value=0.15, 
        value=0.05, 
        step=0.01,
        help="Estimasi persentase data ekstrem yang dianggap sebagai anomali dari keseluruhan dataset."
    )

    # Proses Pemodelan Alur Machine Learning secara Real-Time
    with st.spinner("Model Machine Learning sedang menganalisis data..."):
        # Fitur matriks numerik yang digunakan
        X = df_gempa[['latitude', 'longitude', 'magnitudo', 'kedalaman_km']]
        
        # Inisialisasi dan pelatihan model Isolation Forest
        clf = IsolationForest(contamination=kontaminasi, random_state=42)
        df_gempa['status_anomali'] = clf.fit_predict(X)
        df_gempa['skor_anomali'] = clf.score_samples(X)
        
        # Klasifikasi Pelabelan Evaluasi Objek
        df_gempa['label_anomali'] = df_gempa['status_anomali'].apply(
            lambda x: '🚨 Anomali/Ekstrem' if x == -1 else '🟢 Normal'
        )

    # Output Ringkasan Temuan Deteksi Model
    total_anomali = len(df_gempa[df_gempa['status_anomali'] == -1])
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Total Gempa Terdeteksi Anomali", 
            total_anomali, 
            delta=f"{round(kontaminasi*100)}% dari total data", 
            delta_color="inverse"
        )
    with col2:
        st.metric("Total Gempa Berkategori Normal", len(df_gempa) - total_anomali)

    st.markdown("---")

    # Peta Visualisasi Lapisan Klasifikasi Hasil Model ML
    st.subheader("🗺️ Peta Distribusi Geografis Hasil Deteksi Anomali")
    st.markdown("Titik berwarna merah melambangkan gempa pencilan yang memiliki karakteristik tidak biasa (misal: kedalaman ekstrem, magnitudo sangat besar, atau terjadi di klaster koordinat yang sangat rapat/rentetan gempa susulan).")
    
    fig_ml_map = px.scatter_mapbox(
        df_gempa, 
        lat="latitude", 
        lon="longitude", 
        size="magnitudo",
        color="label_anomali", 
        color_discrete_map={'🚨 Anomali/Ekstrem': '#ef4444', '🟢 Normal': '#10b981'},
        hover_name="lokasi_deskripsi",
        hover_data={"waktu_kejadian": True, "magnitudo": True, "kedalaman_km": True, "skor_anomali": True},
        zoom=3.8, 
        center={"lat": -2.5, "lon": 118.0}, 
        mapbox_style="carto-positron", 
        height=500
    )
    st.plotly_chart(fig_ml_map, use_container_width=True)

    st.markdown("---")

    # Penyajian Matriks Deteksi dalam Bentuk Tabel Data Informasi
    st.subheader("📋 Sampel Data Aktivitas Gempa Anomali")
    df_hanya_anomali = df_gempa[df_gempa['status_anomali'] == -1][
        ['waktu_kejadian', 'lokasi_deskripsi', 'magnitudo', 'kedalaman_km', 'skor_anomali']
    ].sort_values(by='skor_anomali')
    
    st.dataframe(df_hanya_anomali, use_container_width=True)


# ==============================================================================
# 3. SISTEM NAVIGASI MULTI-PAGE (STREAMLIT APP CONTROL)
# ==============================================================================
# Mendefinisikan struktur arsitektur navigasi menu internal aplikasi
pg = st.navigation([
    st.Page(halaman_dashboard, title="Dashboard Spasio-Temporal", icon="🗺️"),
    st.Page(halaman_ml_anomali, title="Model Deteksi Anomali (ML)", icon="🧠")
])

# Eksekusi perayapan halaman aktif
pg.run()

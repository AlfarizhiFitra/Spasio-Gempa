import streamlit as st
import pandas as pd
import plotly.express as px

# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
st.set_page_config(
    page_title="Dashboard Spasio-Temporal Gempa Indonesia",
    page_icon="🌋",
    layout="wide"
)

# 2. FUNGSI UNTUK MEMBACA DATASET
@st.cache_data 
def load_data():
    df = pd.read_csv("dataset_gempa_indonesia_2023_2026.csv")
    df['waktu_kejadian'] = pd.to_datetime(df['waktu_kejadian'])
    return df

# Memuat data ke dalam aplikasi
try:
    df_gempa = load_data()
except FileNotFoundError:
    st.error("File 'dataset_gempa_indonesia_2023_2026.csv' tidak ditemukan. Silakan jalankan skrip akuisisi data terlebih dahulu!")
    st.stop()

# 3. MEMBUAT SIDEBAR UNTUK FILTER SPASIO-TEMPORAL
st.sidebar.header("⚙️ Filter Spasio-Temporal")

# Filter Rentang Waktu (Temporal Slider)
tahun_list = sorted(df_gempa['tahun_periode'].unique())
tahun_terpilih = st.sidebar.select_slider(
    "Pilih Tahun Pengamatan:",
    options=tahun_list,
    value=tahun_list[-1] 
)

# Filter Kekuatan Gempa (Atribut Magnitudo)
min_mag, max_mag = float(df_gempa['magnitudo'].min()), float(df_gempa['magnitudo'].max())
magnitudo_terpilih = st.sidebar.slider(
    "Minimal Magnitudo (SR):",
    min_value=min_mag,
    max_value=max_mag,
    value=4.0, 
    step=0.1
)

# 4. PROSES PENYARINGAN DATA (DYNAMIC FILTERING)
df_filtered = df_gempa[
    (df_gempa['tahun_periode'] == tahun_terpilih) & 
    (df_gempa['magnitudo'] >= magnitudo_terpilih)
]

# 5. KONTEN UTAMA DASHBOARD
st.title("🌋 Dashboard Spasio-Temporal Kebencanaan Gempa Bumi Indonesia")

# Menggunakan raw string 'r' untuk merender ekspresi matematika LaTeX \ge tanpa memicu escape sequence warning
st.markdown(r"Menampilkan aktivitas seismik riil Indonesia pada tahun **{}** dengan kekuatan **$\ge$ {} SR**.".format(tahun_terpilih, magnitudo_terpilih))

# Membuat Ringkasan Metrik Singkat di Atas Halaman
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

# 6. VISUALISASI UTAMA: PETA SPASIAL INTERAKTIF (PLOTLY MAPBOX)
st.subheader("🗺️ Peta Sebaran Episentrum Spasial")

if not df_filtered.empty:
    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="latitude",
        lon="longitude",
        size="magnitudo",          
        color="kedalaman_km",      
        color_continuous_scale="inferno", # 🚀 Perbaikan: Menggunakan string nama palet warna bawaan yang aman
        hover_name="lokasi_deskripsi",
        hover_data={"waktu_kejadian": True, "magnitudo": True, "kedalaman_km": True},
        zoom=3.8,
        center={"lat": -2.5, "lon": 118.0}, 
        mapbox_style="carto-positron",     
        height=550
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Tidak ada aktivitas gempa yang memenuhi kriteria filter pada periode ini.")

st.markdown("---")

# 7. VISUALISASI PENDUKUNG: TREN TEMPORAL (GRAFIK BULANAN)
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
        height=350
    )
    st.plotly_chart(fig_line, use_container_width=True)

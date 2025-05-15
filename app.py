
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# Fungsi: Normalisasi & WP
# ============================
def normalisasi_wp(df, bobot, jenis):
    x = df.iloc[:, 1:].values
    for i in range(x.shape[1]):
        if jenis[i] == 'benefit':
            x[:, i] = x[:, i] / x[:, i].max()
        else:
            x[:, i] = x[:, i].min() / x[:, i]
    return x

def hitung_wp(df, bobot, jenis):
    x_norm = normalisasi_wp(df, bobot, jenis)
    skor = np.prod(x_norm ** bobot, axis=1)
    df["Skor"] = skor
    return df.sort_values(by="Skor", ascending=False)

# ============================
# Load Model Clustering & Bobot
# ============================
@st.cache_resource
def load_model():
    model_cluster = joblib.load("model_cluster.pkl")
    cluster_bobot = joblib.load("cluster_bobot.pkl")
    return model_cluster, cluster_bobot

model_cluster, cluster_bobot = load_model()

# ============================
# Streamlit App
# ============================
st.title("🤖 Sistem Rekomendasi Laptop Berbasis AI")
st.markdown("Menggunakan Machine Learning (Unsupervised Clustering) + Weighted Product")

uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset Laptop", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🗃️ Dataset Laptop")
    st.dataframe(df)

    st.subheader("📋 Input Preferensi Kamu")
    max_harga = st.number_input("Harga Maksimal (Rp)", value=10000000)
    min_ram = st.number_input("Minimal RAM (GB)", value=8)
    min_prosesor = st.number_input("Minimal Skor Prosesor (1-10)", value=5)
    min_storage = st.number_input("Minimal Penyimpanan (GB)", value=256)

    input_user = [[max_harga, min_ram, min_prosesor, min_storage]]
    cluster_id = model_cluster.predict(input_user)[0]
    st.info(f"📌 Kamu termasuk dalam Cluster #{cluster_id}")

    bobot = cluster_bobot[cluster_id]
    bobot = [b / sum(bobot) for b in bobot]  # Normalisasi
    jenis = ['cost', 'benefit', 'benefit', 'benefit']

    df_filter = df[
        (df["Harga"] <= max_harga) &
        (df["RAM"] >= min_ram) &
        (df["Prosesor"] >= min_prosesor) &
        (df["Penyimpanan"] >= min_storage)
    ]

    if df_filter.empty:
        st.warning("❌ Tidak ada laptop yang sesuai.")
    else:
        if st.button("🔍 Rekomendasikan Laptop"):
            hasil = hitung_wp(df_filter.copy(), bobot, jenis)
            st.subheader("📈 Hasil Rekomendasi")
            st.dataframe(hasil[["Nama", "Skor"]])
            st.success(f"🎯 Laptop terbaik: {hasil.iloc[0]['Nama']}")
else:
    st.info("Silakan upload dataset laptop (.csv) terlebih dahulu.")

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("label_encoder.joblib")

# Konfigurasi halaman
st.set_page_config(
    page_title="Spotify Genre Prediction",
    page_icon="🎵",
    layout="wide"
)

# Header
st.title("🎵 Spotify Genre Classification System")
st.markdown("Prediksi genre lagu berdasarkan karakteristik audio Spotify menggunakan Machine Learning.")

st.divider()

# Metric Atas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Model Terbaik",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="📈 Akurasi",
        value="92%"
    )

with col3:
    st.metric(
        label="🎵 Dataset",
        value="114K"
    )

with col4:
    st.metric(
        label="🎼 Genre",
        value="114"
    )

st.divider()

st.subheader("🎧 Masukkan Karakteristik Lagu")

col1, col2 = st.columns(2)

with col1:

    danceability = st.slider(
        "Danceability",
        0.0, 1.0, 0.5
    )

    energy = st.slider(
        "Energy",
        0.0, 1.0, 0.5
    )

    speechiness = st.slider(
        "Speechiness",
        0.0, 1.0, 0.1
    )

    acousticness = st.slider(
        "Acousticness",
        0.0, 1.0, 0.2
    )

with col2:

    instrumentalness = st.slider(
        "Instrumentalness",
        0.0, 1.0, 0.0
    )

    valence = st.slider(
        "Valence",
        0.0, 1.0, 0.5
    )

    tempo = st.slider(
        "Tempo",
        50,
        250,
        120
    )

st.divider()

if st.button(
    "🎵 Prediksi Genre",
    use_container_width=True
):

    input_data = pd.DataFrame([[
        danceability,
        energy,
        speechiness,
        acousticness,
        instrumentalness,
        valence,
        tempo
    ]], columns=[
        'danceability',
        'energy',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'valence',
        'tempo'
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    genre = le.inverse_transform(prediction)[0]

    st.divider()

    st.subheader("🎯 Hasil Prediksi")

    st.success(
        f"🎧 Genre Lagu: {genre}"
    )

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(input_scaled)[0]

        confidence = probs.max() * 100

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

st.divider()

st.caption(
    "Final Project AI & Big Data 2026 | Spotify Genre Classification"
)

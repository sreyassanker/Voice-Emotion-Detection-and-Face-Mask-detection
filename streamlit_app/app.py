import streamlit as st
import librosa
import numpy as np
import sounddevice as sd 
import pickle

# =========================
# LOAD MODEL
# =========================
with open("../voice_emotion/emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(audio, sr=22050):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

def record_audio(duration=3, sr=22050):
    st.info("🎙 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    st.success("Recording complete")
    return audio.flatten(), sr

# =========================
# UI
# =========================
st.title("🎧 Voice Emotion Detection")

if st.button("Record Voice"):
    audio, sr = record_audio()
    features = extract_features(audio, sr)
    prediction = model.predict([features])[0]

    st.subheader(f"🧠 Predicted Emotion: {prediction.upper()}")

    probs = model.predict_proba([features])[0]
    for emo, prob in zip(model.classes_, probs):
        st.write(f"{emo}: {prob:.2f}")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded_file:
    audio, sr = librosa.load(uploaded_file, sr=22050)
    features = extract_features(audio, sr)
    prediction = model.predict([features])[0]

    st.subheader(f"🧠 Predicted Emotion: {prediction.upper()}")

import os
import zipfile
import pickle
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# CONFIG
# =========================
DATA_ZIP = "emotion_detection.zip"
EXTRACT_DIR = "dataset"

emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# =========================
# LOAD DATA
# =========================
X, y = [], []

for root, _, files in os.walk(EXTRACT_DIR):
    for file in tqdm(files):
        if file.endswith(".wav"):
            try:
                emotion_code = file.split("_")[2]
                emotion = emotion_map.get(emotion_code)
                if emotion:
                    features = extract_features(os.path.join(root, file))
                    X.append(features)
                    y.append(emotion)
            except Exception as e:
                print(f"Skipping {file}: {e}")

X = np.array(X)
y = np.array(y)

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# =========================
# SAVE MODEL
# =========================
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Emotion model saved")

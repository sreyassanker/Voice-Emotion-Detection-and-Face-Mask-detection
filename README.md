# 🎭 Voice Emotion & Face Mask Detection System

A dual AI system combining **speech emotion recognition** and **real-time face mask detection** using deep learning and computer vision.

## 🔹 Modules
- **Voice Emotion Detection** 
  - MFCC, Chroma, Mel features
  - Random Forest classifier
  - Real-time prediction via Streamlit

- **Face Mask Detection**
  - CNN-based classifier
  - OpenCV real-time face detection
  - Audio alert for no-mask detection

## 🛠 Tech Stack
- Python, NumPy, OpenCV
- TensorFlow / Keras
- Librosa (Audio Processing)
- Flask & Streamlit

## ▶ How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
python face_mask/app.py

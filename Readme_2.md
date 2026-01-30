# 📘 Technical Documentation  
## Voice Emotion & Face Mask Detection System

This document provides a **detailed, process-level explanation** of each module, including data flow, feature extraction, model training, and real-time inference logic.

--- 

## 🧩 Project Architecture Overview

The system consists of **two independent AI pipelines**:

1. **Voice Emotion Detection Pipeline**
2. **Real-Time Face Mask Detection Pipeline**

Each pipeline is modular, loosely coupled, and deployable independently.

---

## 📁 Directory Structure Explained


### 🎯 Objective
Classify human emotions from speech signals using **handcrafted audio features** and a **machine learning classifier**.

---

### 🔹 Dataset Handling

- Input: `.wav` audio files
- File naming convention encodes emotion labels (e.g. `ANG`, `HAP`, `SAD`)
- Audio clips are truncated/padded to **3 seconds** for consistency

---

### 🔹 Feature Extraction Pipeline

For each audio sample:

1. **MFCC (Mel Frequency Cepstral Coefficients)**
   - Captures vocal tract characteristics
2. **Chroma Features**
   - Represents pitch class distribution
3. **Mel Spectrogram**
   - Encodes perceptual frequency energy

All features are:
- Mean-aggregated across time
- Concatenated into a single feature vector

```text
Final Feature Vector = [MFCC | Chroma | Mel]

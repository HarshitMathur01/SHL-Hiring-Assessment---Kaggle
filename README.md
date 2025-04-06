# 🧠 SHL Grammar Scoring - Kaggle Competition 2025

This repository contains my complete solution for the **SHL Hiring Assessment** hosted on Kaggle. The task is to predict grammar proficiency scores (0 to 5) from audio clips of spoken English by candidates. I explored multiple deep learning and machine learning approaches using both audio signals and transcribed text.

> **Final Best Approach:** A **multi-modal ensemble** combining audio features + Whisper transcripts, fed into an **MLP head**.

---

## 📌 Problem Overview

Given a dataset of audio responses and grammar scores:
- Predict a continuous grammar score for new audio clips.
- Evaluation Metric: **Mean Squared Error (MSE)**

---

## 🗂️ Dataset

Each training sample includes:
- `.wav` audio file (spoken answer)
- `label` (grammar proficiency, float between 0 and 5)

---

## 🔍 Approaches Overview

### 🔹 **Approach 1: Audio Features → XGBoost / MLP**
- Used `librosa` to extract:
  - Waveform 
- Models:
  - `facebook/wav2vec2-base-960h` for audio features
  - `XGBoostRegressor` with `RandomizedSearchCV` for tuning
  - Deep `MLP Regressor` for required output
- Insights:
  - Fast to compute but limited by shallow semantics

---

### 🔹 **Approach 2: Transcript → Text-based Regression**
- Used **Whisper** to transcribe audio to text
- Processed text with:
  - BERT tokenizer + embeddings (`bert-base-uncased`)
- Fed into:
  - MLP
- Strength: Captured syntactic and grammatical errors well

---

### 🔹 **Approach 3: Audio Embeddings from Wav2Vec2** + Text fusision
- Used **facebook/wav2vec2-base-960h** to extract embeddings from raw waveforms
- Pros:
  - Learned rich acoustic representations

---

### ✅ **Approach 4 (Best): Audio + Text Fusion**
- Combined:
  - WaveLM audio embeddings
  - Whisper transcripts → BERT embeddings
- Concatenated into a single feature vector
- Fed into a custom **MLP regressor**
- Result: **Lowest MSE on validation set**

---

## 🧪 Model Architecture

```text
Input (Audio Features + Text Embeddings)
             ↓
          Concatenation
             ↓
         BatchNorm1d
             ↓
            MLP
        (ReLU + Dropout)
             ↓
          Linear Out
             ↓
        Grammar Score

# Violence_detection




# 🎬 Violence Detection in Videos using CNN + LSTM

## 📌 Overview

This project detects **violent activities in videos** using a deep learning pipeline that combines:

* **CNN (MobileNetV2)** for spatial feature extraction
* **LSTM** for temporal sequence learning

The model processes video frames in sequences and predicts whether a segment contains violence or not.

---

## 🚀 Features

* 🎥 Video-based inference (not just images)
* 🧠 CNN + LSTM hybrid architecture
* 🔁 Sliding window prediction for continuous detection
* ⚡ Efficient feature extraction pipeline
* 📊 Confidence score output for each sequence

---

## 🧱 Project Architecture

```
Video → Frame Extraction → Preprocessing → CNN (Feature Extractor)
      → Feature Sequences → LSTM → Prediction (Violent / Normal)
```

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

---

## 📂 Project Structure

```
├── data/
│   ├── violence/
│   └── non_violence/
├── models/
│   ├── cnn_extractor.h5
│   └── lstm_model.h5
├── utils/
│   ├── preprocessing.py
│   └── data_loader.py
├── train.py
├── predict.py
├── config.py
└── README.md
```

---

## ⚙️ How It Works

### 1. Frame Extraction

* Videos are split into frames using OpenCV
* Frames are resized and normalized

### 2. Feature Extraction (CNN)

* Pretrained MobileNetV2 extracts features from each frame
* Converts images → feature vectors

### 3. Sequence Modeling (LSTM)

* Sequences of frame features are passed to LSTM
* Learns temporal patterns (motion, changes)

### 4. Prediction

* Outputs probability of violence
* Uses threshold (default = 0.5)

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install tensorflow opencv-python numpy
```

### 2. Train Model

```bash
python train.py
```

### 3. Run Prediction

```bash
python predict.py --video path/to/video.mp4
```

---

## 🧪 Inference Logic

* Frames are processed in sequences (e.g., 20 frames)
* Sliding window approach (overlapping sequences)
* Each window produces a prediction

---

## 📊 Output Example

```
🎬 Analyzing Video: sample.mp4

Result: ⚠️ VIOLENCE DETECTED | Confidence: 0.87
Result: ✅ NORMAL | Confidence: 0.21
```

---

## ⚠️ Challenges Faced

* ⏳ Slow training due to repeated frame extraction
* 💾 Memory issues when caching large datasets
* ⚖️ Balancing dataset (violence vs non-violence)

---

## 💡 Improvements (Future Work)

* 🔥 Real-time detection using webcam
* 📦 Model optimization (quantization / pruning)
* 🎯 Better dataset for higher accuracy
* 🖼️ Add bounding boxes for detected actions
* ⚡ GPU acceleration for faster inference

---

## 🤝 Contribution

Feel free to fork the repo and improve the model or pipeline.

---

## 📜 License

This project is for educational purposes.

---

## 👨‍💻 Author

Aman

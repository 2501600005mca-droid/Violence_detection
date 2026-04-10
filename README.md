
# 🛡️ Real-Time Violence Detection System (CNN + LSTM)

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A deep learning project designed to detect violent activities in real-world CCTV and video footage. This system utilizes a hybrid **CNN-LSTM architecture** to analyze spatial features and temporal sequences, ensuring high accuracy and real-time inference capabilities.

---

## 📌 Overview
Detecting violence in surveillance videos requires understanding both the visual elements (spatial) and the sequence of actions over time (temporal). 
This project solves this by:
1. **Spatial Feature Extraction:** Using a pre-trained **MobileNetV2** (CNN) to extract a 1280-dimensional feature vector from individual video frames.
2. **Temporal Modeling:** Using **Long Short-Term Memory (LSTM)** networks to analyze the sequence of these features over 20 frames to classify the video segment as `Violent` or `Normal`.

## 🏗️ Architecture Pipeline
```text
Video File (.mp4) 
 ├──> Extract 20 Frames (OpenCV) 
 │     ├──> Preprocess & Resize (224x224) 
 │     │     ├──> MobileNetV2 (Feature Extraction) 
 │     │     │     ├──> 1280-dim feature vectors (.npy)
 │     │     │     │     ├──> LSTM Layers (Temporal Learning)
 │     │     │     │     │     └──> Dense + Sigmoid Layer = Probability Output (0 to 1)
```

## 🚀 Key Features
* **Lightning Fast Training:** Implemented an "Offline Feature Extraction" pipeline. Video frames are processed through MobileNetV2 once and saved as `.npy` files. The LSTM is trained directly on these numerical features, reducing epoch times from minutes to mere seconds.
* **Memory Efficient:** Uses custom `tf.keras.utils.Sequence` Data Generators to load batches of data dynamically, preventing RAM crashes on large datasets.
* **High Accuracy:** Achieved **~99% Training Accuracy** and **~97% Validation Accuracy** with a robust AUC score of 0.99.
* **Real-Time Inference:** The prediction engine uses a sliding window approach to analyze ongoing video streams efficiently.

## 📂 Dataset
The model was trained on the [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) from Kaggle, which contains 2000 videos (1000 violent, 1000 non-violent) collected from YouTube.

## 💻 Installation & Setup
This project is highly optimized for Google Colab and local environments with GPU support.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Violence-Detection-LSTM.git](https://github.com/your-username/Violence-Detection-LSTM.git)
   cd Violence-Detection-LSTM
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow opencv-python numpy scikit-learn tqdm kagglehub
   ```

## ⚙️ How to Run

### 1. Training the Model
Open the provided Jupyter Notebook (`Final_Violence_Detection.ipynb`) and run the cells sequentially. 
* Phase 1 will download the dataset and extract `.npy` features.
* Phase 2 will train the LSTM network on the extracted features.
* The best model is automatically saved as a `.keras` file using the `ModelCheckpoint` callback.

### 2. Running Inference on Custom Videos
Once trained, you can test the model on your own videos using the `predict_video` function:

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Load the trained model and feature extractor
model = tf.keras.models.load_model("path/to/final_violence_model_v1.keras")
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Run prediction
predict_video("path/to/test_video.mp4", model, feature_extractor, CONFIG)
```

**Output Example:**
```text
🎬 Analyzing Video: test_video.mp4
Result: ✅ NORMAL | Confidence: 0.0004
Result: ⚠️ VIOLENCE DETECTED | Confidence: 0.9986
Result: ⚠️ VIOLENCE DETECTED | Confidence: 0.9997
```

## 📈 Model Performance
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam (Learning Rate: 1e-4 with ReduceLROnPlateau)
* **Regularization:** Batch Normalization & Dropout (0.4) to prevent overfitting.
* **Validation AUC:** 0.9981

## 🔮 Future Enhancements
- [ ] Implement audio feature extraction for multimodal detection (shouting, glass breaking).
- [ ] Integrate with a real-time webcam feed.
- [ ] Send automated email/SMS alerts upon violence detection.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Violence-Detection-LSTM/issues).

---
*Built as an MCA Academic Project — Bridging the gap between AI and Public Safety.*

https://github.com/your-username/Violence-Detection-LSTM.git

# ThoraxScan AI: Explainable Respiratory Disease Detection 🫁

ThoraxScan AI is an end-to-end computer vision and deep learning web platform designed to classify Chest X-rays into four distinct diagnostic categories (**Normal**, **COVID-19**, **Pneumonia**, and **Tuberculosis**). 

The platform incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to eliminate the "black-box" nature of neural networks, providing clinicians with clear visual heatmaps of highlighted anatomical abnormalities.

---

## 🌟 Key Features
- **Multi-Class Pathology Detection:** Classifies input X-rays across 4 target categories.
- **Explainable AI (XAI):** Real-time Grad-CAM heatmap visualization to verify infected lung zones.
- **Image Enhancement Pipeline:** Integrates OpenCV **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for localized contrast boosting.
- **Lightweight & High Speed:** Parameter-efficient DenseNet-121 backbone enabling sub-second inference on standard CPU hardware.

---

## 🛠️ Tech Stack
- **Language & Framework:** Python 3.9+, TensorFlow / Keras
- **Computer Vision:** OpenCV, Pillow, Matplotlib
- **Web Application:** Streamlit
- **Deployment:** Streamlit Community Cloud

---

## 🚀 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/rambuilds08/MiniProject.git](https://github.com/rambuilds08/MiniProject.git)
   cd MiniProject

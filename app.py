import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import matplotlib.cm as cm
import gdown



# ==========================================
# PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(
    page_title="ThoraxScan AI - Diagnostic Triage",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 ThoraxScan AI: Respiratory Disease Triage & Explainability")
st.markdown("""
**Educational Research & Decision Support System**  
An explainable deep learning pipeline for multi-class classification of Chest X-rays (COVID-19, Pneumonia, Tuberculosis, Normal) utilizing **DenseNet-121** and **Grad-CAM visual heatmaps**.
""")
st.divider()

# ==========================================
# MODEL CONFIGURATION & CACHED LOAD
# ==========================================

CLASSES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]



MODEL_PATH = "Xscan_real_model.h5"
# Replace with your actual Google Drive File ID
GDRIVE_FILE_ID = "1dXveu4UoHLeFfddmAxkITCHTYOGUIn3U"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model weights (first-time startup)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model binary: {e}")
        return None

model = load_trained_model()

if model is None:
    st.error(f"⚠️ **Model File Missing (`{MODEL_PATH}`)**")
    st.info("Please upload `Xscan_real_model.h5` to the root directory of your GitHub repository to enable live inference.")
    st.stop()

# ==========================================
# IMAGE PROCESSING & GRAD-CAM PIPELINE
# ==========================================
def preprocess_image(img_arr):
    # Grayscale conversion using OpenCV
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_arr
        
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3-channel scaling for DenseNet backbone & normalization to [0, 1]
    rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized_img = cv2.resize(rgb_img, (224, 224))
    normalized_img = resized_img.astype(np.float32) / 255.0
    return np.expand_dims(normalized_img, axis=0), resized_img

def compute_gradcam(model, processed_tensor, target_layer_name=None):
    if target_layer_name is None:
        # Auto-detect final convolutional layer for DenseNet / ResNet architectures
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Concatenate)):
                target_layer_name = layer.name
                break
        if not target_layer_name:
            target_layer_name = model.layers[-4].name

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_tensor)
        top_pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, top_pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap safely
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colormap_colors = colormap(np.arange(256))[:, :3]
    jet_heatmap = colormap_colors[heatmap_uint8]
    
    jet_heatmap = cv2.resize(jet_heatmap, (original_img.shape[1], original_img.shape[0]))
    overlay = jet_heatmap * alpha + (original_img / 255.0)
    return np.clip(overlay, 0, 1)

# ==========================================
# INTERACTIVE DIAGNOSTIC DASHBOARD
# ==========================================
uploaded_file = st.file_uploader(
    "Upload Patient Chest X-Ray Scan (PNG, JPG, JPEG)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(raw_image)
    
    tensor_input, resized_rgb = preprocess_image(img_array)
    
    with st.spinner("Executing DenseNet-121 feature extraction & Grad-CAM visual synthesis..."):
        predictions = model.predict(tensor_input)
        pred_idx = int(np.argmax(predictions[0]))
        confidence_score = float(predictions[0][pred_idx]) * 100
        predicted_class = CLASSES[pred_idx]
        
        # Generate Explainability Overlay
        heatmap = compute_gradcam(model, tensor_input)
        heatmap_overlay = overlay_heatmap(resized_rgb, heatmap)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Input Chest X-Ray")
        st.image(raw_image, use_container_width=True)
        
    with col2:
        st.subheader("🔥 Grad-CAM Pathology Heatmap")
        st.image(heatmap_overlay, use_container_width=True, clamp=True)
        st.caption("Warm regions (Red/Yellow) highlight anatomical patterns driving the network's classification.")

    st.markdown("---")
    st.subheader("📊 Diagnostic Summary & Multi-Class Confidence")
    
    if predicted_class == "Normal":
        st.success(f"**Predicted Diagnosis:** {predicted_class} (Confidence: {confidence_score:.2f}%)")
    else:
        st.error(f"**Predicted Diagnosis:** {predicted_class} (Confidence: {confidence_score:.2f}%)")
        
    st.write("### Class Probability Breakdown:")
    for idx, class_name in enumerate(CLASSES):
        prob = float(predictions[0][idx])
        st.progress(prob, text=f"{class_name}: {prob*100:.2f}%")

st.sidebar.markdown("""
### About ThoraxScan AI
* **Architecture:** DenseNet-121 (Transfer Learning)
* **Pre-processing:** OpenCV CLAHE Grayscale Optimization
* **Explainability:** Grad-CAM (XAI)
* **Goal:** Triage assistance for resource-constrained clinics.

*Disclaimer: For research and academic demo purposes only.*
""")

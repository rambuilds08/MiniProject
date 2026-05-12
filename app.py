import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. Load the AI Brain
@st.cache_resource
def load_xscan_model():
    return tf.keras.models.load_model('Xscan_real_model.h5')

model = load_xscan_model()



def make_gradcam_heatmap(img_array, model):
    with tf.GradientTape() as tape:
        x = img_array
        
        densenet_layer = model.get_layer('densenet121')
        densenet_idx = model.layers.index(densenet_layer)
        
        # Walk through early layers (Rescaling, Augmentation)
        for layer in model.layers[1:densenet_idx]: # Skip the input layer at index 0
            if "augmentation" in layer.name:
                x = layer(x, training=False) # Tell augmentation NOT to flip/zoom
            else:
                x = layer(x) # Rescaling just takes the input naturally
            
        # Pass through DenseNet to get the final feature map
        last_conv_output = densenet_layer(x, training=False)
        tape.watch(last_conv_output)
        
        # Pass through the remaining top layers
        preds = last_conv_output
        for layer in model.layers[densenet_idx+1:]:
            preds = layer(preds)
            
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradients!
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Apply the gradients to the feature map to create the heatmap
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    # Load original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Superimpose
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# CRITICAL FIX: These must match the alphabetical order of your exact folder names
# We map them to pretty display names for the UI.
folder_names = ['covid', 'normal', 'pnemonia', 'tb']
display_classes = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# 2. Calculation Logic
# 2. Calculation Logic (Updated for Visibility)
def calculate_system_metrics(test_ds):
    y_true = []
    y_pred = []
    y_probs = []

    # Total number of images
    num_samples = test_ds.samples
    # Create a progress bar in the Streamlit UI
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    # We iterate through the generator manually to track progress
    # batch_size=1 makes this easier to track 1-by-1
    for i in range(num_samples):
        # Get next image and label
        img, label = next(test_ds)
        
        # Predict (verbose=1 will now show progress in PowerShell)
        preds = model.predict(img, verbose=1) 
        
        y_probs.extend(preds)
        y_pred.append(np.argmax(preds, axis=1)[0])
        y_true.append(np.argmax(label, axis=1)[0])
        
        # Update progress bar in browser
        percent_complete = (i + 1) / num_samples
        my_bar.progress(percent_complete, text=f"Analyzing Image {i+1} of {num_samples}...")

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate final stats
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=4)
    mse = mean_squared_error(y_true_oh, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    # Remove the progress bar when done
    my_bar.empty()

    return acc, prec, rec, mse, cm

# 3. UI Layout
st.set_page_config(page_title="XScan AI | Research Analytics", layout="wide")
st.title("XScan AI")
st.subheader("Multi-Class Respiratory Disease Detection System")

# --- Performance Calculation & Graph Section ---
st.sidebar.header("Model Validation")
if st.sidebar.button("Run System Validation"):
    with st.spinner("Analyzing test directory & generating graphs..."):
        try:
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            
            # Make sure this path points to the 'test' folder in your screenshot
            test_generator = test_datagen.flow_from_directory(
                'test', 
                target_size=(224, 224),
                batch_size=1,
                class_mode='categorical',
                shuffle=False # MUST be False to keep true labels aligned with predictions
            )

            acc, prec, rec, mse, cm = calculate_system_metrics(test_generator)

            # Display the calculated metrics
            st.markdown("### 📊 Calculated Performance Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.2%}")
            c2.metric("Precision", f"{prec:.2%}")
            c3.metric("Recall", f"{rec:.2%}")
            c4.metric("MSE", f"{mse:.4f}")
            
            st.markdown("---")
            
            # Display the Confusion Matrix Graph
            st.markdown("### 📉 Confusion Matrix Result Analysis")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=display_classes, 
                        yticklabels=display_classes, 
                        ax=ax)
            plt.ylabel('Actual Diagnosis', fontsize=12)
            plt.xlabel('AI Predicted Diagnosis', fontsize=12)
            plt.title('XScan AI Classification Results', fontsize=14)
            
            # Render graph in Streamlit
            st.pyplot(fig)
           

        except Exception as e:
            st.error(f"Error loading test set: {e}. Make sure the 'test' folder is in the same directory as this script.")

st.markdown("---")

## 4. Inference Section (Single Image Upload)
st.markdown("---")
st.markdown("### 🩺 Live Patient Scan")
uploaded_file = st.file_uploader("Upload a Chest X-Ray for Live Analysis", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Display the original uploaded image
    st.image(uploaded_file, caption="Uploaded Chest X-Ray", use_container_width=True)
    
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing scan and generating heatmap..."):
          # Prepare image array
            img = Image.open(uploaded_file).convert('RGB')
            img = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Remove the / 255.0! The new model handles it internally.
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            confidence = np.max(predictions) * 100
            classes = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
            predicted_class = classes[np.argmax(predictions)]
            
            # Generate Heatmap
            # (We temporarily remove softmax to get raw gradients, then put it back)
           # Generate Heatmap
            # (We temporarily remove softmax to get raw gradients, then put it back)
            model.layers[-1].activation = None
            
            # UPDATE: Tell Grad-CAM to target the nested base model layer!
            heatmap = make_gradcam_heatmap(img_array, model)
            
            model.layers[-1].activation = tf.keras.activations.softmax
            
            # ... (your existing prediction and heatmap generation code) ...
            
            final_img = overlay_heatmap(uploaded_file, heatmap)

            # --- DISPLAY RESULTS ---
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🩺 Diagnosis: **{predicted_class}**")
                st.info(f"**Confidence Score:** {confidence:.2f}%")
                
                with st.expander("View detailed probabilities"):
                    for i, class_name in enumerate(classes):
                        st.write(f"- {class_name}: {predictions[0][i] * 100:.2f}%")
                        
            with col2:
                # 1. Create a matplotlib figure
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # 2. Display the superimposed image on the axis
                ax.imshow(final_img)
                ax.axis('off') # Hide the standard chart x/y axes
                
                # 3. Build the standalone colorbar
                norm = mcolors.Normalize(vmin=0, vmax=1)
                sm = cm.ScalarMappable(cmap='jet', norm=norm)
                sm.set_array([])
                
                # 4. Attach the colorbar to the side and format the labels
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks([0, 0.5, 1])
                cbar.set_ticklabels(['Low', 'Medium', 'High'])
                cbar.set_label('Pathology Activation Intensity', rotation=270, labelpad=15)
                
                # 5. Render the entire figure in Streamlit (UPDATED LINE)
                st.pyplot(fig, use_container_width=True)
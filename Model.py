import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.models import Model
from google.colab import drive

# --- 1. ENVIRONMENT SETUP ---
print("Connecting to Google Drive...")
drive.mount('/content/drive')

print("Extracting dataset ...")
!unzip -q -o "/content/drive/MyDrive/Xscan_Dataset.zip" -d "/content/"

dataset_path = '/content/Xscan_Dataset'
img_size = (224, 224)
batch_size = 32

# --- 2. LOAD DATA ---
print("Loading images into training pipeline...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Classes successfully detected: {class_names}")

# --- 3. CALCULATE CLASS WEIGHTS (For Imbalance) ---
print("Analyzing dataset imbalance and calculating class weights...")
# Extract labels from the training dataset
train_labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)
unique_classes = np.unique(train_labels)

# Compute mathematically balanced weights
weights = compute_class_weight(
    class_weight='balanced', 
    classes=unique_classes, 
    y=train_labels
)
class_weights_dict = dict(zip(unique_classes, weights))

print("Applied Class Weights:")
for i, c_name in enumerate(class_names):
    print(f" - {c_name}: {class_weights_dict[i]:.2f}")

# --- 4. PERFORMANCE CACHING ---
print("Applying AUTOTUNE caching for faster training...")
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. BUILD AUGMENTATION & ARCHITECTURE ---
print("Constructing XScan AI with Augmentation & DenseNet121...")

# Define medical-grade data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
], name="medical_augmentation")

# Input Layer
inputs = Input(shape=(224, 224, 3))

# Pixel Rescaling (Moved inside the model for safer deployment)
x = Rescaling(1./255)(inputs)

# Augmentation (Automatically turns off during validation/testing)
x = data_augmentation(x)

# Base Model (DenseNet121)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model(x, training=False)

# Custom Top Layers
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Finalize and Compile
model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. EXECUTE TRAINING ---
print("Starting 10-Epoch Training Sequence...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights_dict  # <-- Injects the imbalance protection
)

# --- 7. SAVE MODEL ---
model.save('/content/drive/MyDrive/Xscan_real_model.h5')
print("TRAINING COMPLETE: Robust Model saved to your Google Drive as 'Xscan_real_model.h5'")

---

#### 4. `training/model_training.py`
```python
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.metrics as metrics

# ==========================================
# 1. PRE-PROCESSING & CLAHE PIPELINE
# ==========================================
def apply_clahe_preprocessing(img):
    img_uint8 = np.array(img, dtype=np.uint8)
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb_img.astype(np.float32)

# ==========================================
# 2. DATA AUGMENTATION & GENERATORS
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "dataset/"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    preprocessing_function=apply_clahe_preprocessing,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 3. DENSENET-121 ARCHITECTURE
# ==========================================
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=[
        metrics.CategoricalAccuracy(name='accuracy'),
        metrics.Recall(name='recall'),
        metrics.Precision(name='precision')
    ]
)

# ==========================================
# 4. CALLBACKS & TRAINING
# ==========================================
callbacks = [
    ModelCheckpoint('Xscan_real_model.h5', save_best_only=True, monitor='val_recall', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    EarlyStopping(monitor='val_recall', patience=6, restore_best_weights=True)
]

if __name__ == "__main__":
    print("Beginning DenseNet-121 Model Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=callbacks
    )
    print("Training complete. Binary saved as Xscan_real_model.h5")

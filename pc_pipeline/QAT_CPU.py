import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# =============================
# Config
# =============================
IMG_SIZE = (160, 160)
BATCH_SIZE = 8        # CPU 可接受
EPOCHS = 6            # QAT 建議 >5
LEARNING_RATE = 1e-4  # 一定要小
FP32_MODEL = "best_pet_classifier.h5"

DATA_DIR = "resized"
VAL_DIR = "validation_backup"

# =============================
# Data
# =============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =============================
# Load FP32 model
# =============================
base_model = tf.keras.models.load_model(FP32_MODEL)

# =============================
# Apply QAT
# =============================
qat_model = tfmot.quantization.keras.quantize_model(base_model)

qat_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =============================
# QAT Training (CPU)
# =============================
print("\n=== QAT Fine-tuning on CPU ===")
qat_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# =============================
# Export QAT model
# =============================
qat_model.save("qat_pet_classifier_cpu.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_qat = converter.convert()

with open("pet_classifier_int8_QAT_CPU.tflite", "wb") as f:
    f.write(tflite_qat)

print("✔ Saved QAT INT8 model (CPU): pet_classifier_int8_QAT_CPU.tflite")

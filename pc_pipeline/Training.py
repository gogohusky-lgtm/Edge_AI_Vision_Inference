import os
import random
import shutil
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 在檔案最上方（import tensorflow as tf 之後）加入：

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("Enabled GPU memory growth for", len(gpus), "GPU(s).")
    except Exception as e:
        print("Could not set memory growth:", e)

# ==========================================
# STEP 1 — 抽出 10% validation (move)
# ==========================================
DATA_DIR = "resized"
VAL_DIR = "validation_backup"
VAL_RATIO = 0.10

# os.makedirs(VAL_DIR, exist_ok=True)

# print("\n=== STEP 1: Moving 10% validation data ===")
# for cls in os.listdir(DATA_DIR):
#     src_cls = os.path.join(DATA_DIR, cls)
#     if not os.path.isdir(src_cls):
#         continue

#     images = [f for f in os.listdir(src_cls)
#               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     num_total = len(images)
#     num_val = max(1, int(num_total * VAL_RATIO))

#     val_images = random.sample(images, num_val)

#     tgt_cls = os.path.join(VAL_DIR, cls)
#     os.makedirs(tgt_cls, exist_ok=True)

#     for img in val_images:
#         shutil.move(os.path.join(src_cls, img),
#                     os.path.join(tgt_cls, img))

#     print(f"✔ {cls}: moved {num_val}/{num_total} images to validation_backup")

# print("\n=== Validation data prepared! ===\n")


# ==========================================
# STEP 2 — 訓練模型 (90% 資料)
# ==========================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 12
MODEL_PATH = "best_pet_classifier.h5"

# 建立訓練資料集（不再使用 validation_split）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 儲存 class_indices
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("Class indices:", train_generator.class_indices)

import numpy as np

# 計算每一類的資料數量（train_generator.labels 為 numpy array）
unique, counts = np.unique(train_generator.labels, return_counts=True)
class_counts = dict(zip(unique, counts))

num_cat   = int(class_counts.get(0, 0))
num_dog   = int(class_counts.get(1, 0))
num_other = int(class_counts.get(2, 0))

print("Class sample counts:", class_counts)

# 防止某類為 0 而造成除以 0
total = num_cat + num_dog + num_other
n_classes = 3
eps = 1e-6

class_weights = {
    0: total / (n_classes * (num_cat + eps)),
    1: total / (n_classes * (num_dog + eps)),
    2: total / (n_classes * (num_other + eps)),
}

print("\nClass weights:", class_weights)


# 建立模型
base_model = MobileNetV2(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)
model = Model(base_model.input, output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max"),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

print("\n=== STEP 2: Training Model ===")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

print(f"\n✔ Saved best model to {MODEL_PATH}")

# ==========================================
# STEP 2.3 — FP32 TFLite Conversion (Baseline)
# ==========================================

print("\n=== STEP 2.3: FP32 TFLite Conversion ===")

converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter_fp32.convert()

with open("pet_classifier_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)

print("✔ Saved FP32 model: pet_classifier_fp32.tflite")


# ==========================================
# STEP 2.5 — FP16 TFLite Conversion
# ==========================================

print("\n=== STEP 2.5: FP16 TFLite Conversion ===")

converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]

tflite_fp16 = converter_fp16.convert()

with open("pet_classifier_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)

print("✔ Saved FP16 model: pet_classifier_fp16.tflite")



# ==========================================
# STEP 3 — PTQ (INT8)
# ==========================================

print("\n=== STEP 3: PTQ INT8 ===")

def representative_dataset():
    for imgs, _ in val_generator:
        yield [imgs.astype(np.float32)]
        break

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_ptq = converter.convert()

with open("pet_classifier_int8_PTQ.tflite", "wb") as f:
    f.write(tflite_ptq)

print("✔ Saved PTQ model: pet_classifier_int8_PTQ.tflite")


# # ==========================================
# # STEP 4 — QAT 訓練量化
# # ==========================================

# print("\n=== STEP 4: Quantization Aware Training (QAT) ===")

# import tensorflow_model_optimization as tfmot


# # -----------------------------------------
# # 建立小 batch 的 generator 供 QAT 使用
# # -----------------------------------------
# BATCH_SIZE_QAT = 8  # try 8, if OOM -> 4
# train_generator_qat = train_datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE_QAT,
#     class_mode='categorical',
#     shuffle=True
# )
# val_generator_qat = val_datagen.flow_from_directory(
#     VAL_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE_QAT,
#     class_mode='categorical',
#     shuffle=False
# )

# # -----------------------------------------
# # QAT 訓練（使用小 batch）
# # -----------------------------------------

# qat_model = tfmot.quantization.keras.quantize_model(model)
# qat_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# EPOCHS_QAT = 3
# try:
#     qat_model.fit(
#         train_generator_qat,
#         validation_data=val_generator_qat,
#         epochs=EPOCHS_QAT
#     )
# except tf.errors.ResourceExhaustedError as e:
#     print("OOM during QAT. Consider reducing BATCH_SIZE_QAT or run QAT on CPU.")
#     raise

# # 若真的一直 OOM，可在程式最開頭用：
# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 強制在 CPU 上執行

# qat_model.save("qat_pet_classifier.h5")

# # QAT → TFLite
# converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# tflite_qat = converter.convert()
# with open("pet_classifier_int8_QAT.tflite", "wb") as f:
#     f.write(tflite_qat)

# print("✔ Saved QAT INT8 model: pet_classifier_int8_QAT.tflite")


print("\n=== 全流程完成！已輸出模型：===\n")
print(" - best_pet_classifier.h5")
print(" - pet_classifier_int8_PTQ.tflite")
print(" - qat_pet_classifier.h5")
# print(" - pet_classifier_int8_QAT.tflite")


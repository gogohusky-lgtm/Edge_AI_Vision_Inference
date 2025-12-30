import os
import time
import numpy as np
import psutil
import cv2
import tflite_runtime.interpreter as tflite
import lgpio               # 改用 lgpio
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import random

# ======================
# GPIO CONFIG (lgpio)
# ======================
CHIP = 0  # 通常 /dev/gpiochip0
h = lgpio.gpiochip_open(CHIP)

LED_PINS = {
    "cats": 16,
    "dogs": 20,
    "others": 21
}

# 設定為輸出並預設 LOW
for pin in LED_PINS.values():
    lgpio.gpio_claim_output(h, pin)
    lgpio.gpio_write(h, pin, 0)

# ======================
# CONFIG
# ======================
IMG_SIZE = 160
N_RUNS = 20
CLASS_NAMES = ["cats", "dogs", "others"]

MODELS = {
    "FP32": "pet_classifier_fp32.tflite",
    "FP16": "pet_classifier_fp16.tflite",
    "INT8_PTQ": "pet_classifier_int8_PTQ.tflite",
    "INT8_QAT": "pet_classifier_int8_QAT_CPU.tflite",
}

DATASET_DIR = "validation_backup"

# ======================
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
###    img = img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)

# ======================
def show_image(img, pred_label, model_name):
    # 若是 (1, H, W, C)，先拿掉 batch 維度
    if img.ndim == 4:
        img = img[0]

    # 若是 0~1 的 float，轉回 0~255 顯示較正常
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f"{model_name} → Pred: {pred_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ======================
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# ======================
def run_inference(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    # ✅ 確保 batch 維度只出現一次
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    # === Input dtype handling ===
    if input_dtype == np.float32:
        # FP32 / FP16
        input_data = img.astype(np.float32) / 255.0

    elif input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]

        # 🔑 關鍵：QAT 常見 scale = 0 或 1
        if scale == 0 or scale == 1.0:
            # QAT：直接餵 uint8 原始影像
            input_data = img.astype(np.uint8)
        else:
            # PTQ：需依 quantization 參數轉換
            input_data = (img / scale + zero_point).astype(np.uint8)

    else:
        raise TypeError(f"Unsupported input dtype: {input_dtype}")

    # === Inference ===
    interpreter.set_tensor(input_index, input_data)

    start = time.perf_counter()
    interpreter.invoke()
    latency = (time.perf_counter() - start) * 1000  # ms

    output = interpreter.get_tensor(output_details[0]["index"])
    pred = int(np.argmax(output))

    return pred, latency

# ======================
def gpio_output(label):
    for k, pin in LED_PINS.items():
        lgpio.gpio_write(h, pin, 1 if k == label else 0)

# ======================
def benchmark_model(name, model_path, image_paths):
    interpreter = load_model(model_path)

    latencies = []
    correct = 0
    shown = False

    # ✅ 隨機抽 N_RUNS 張（不重複）
    sampled_paths = random.sample(image_paths, N_RUNS)

    cpu_before = psutil.cpu_percent(interval=None)

    for path in sampled_paths:
        true_label = os.path.basename(os.path.dirname(path))
        img = preprocess_image(path)

        pred, latency = run_inference(interpreter, img)
        latencies.append(latency)

        if CLASS_NAMES[pred] == true_label:
            correct += 1

        gpio_output(CLASS_NAMES[pred])

        # 🖼️ 只顯示一次（已修好 batch 問題）
        if not shown:
            show_image(img, CLASS_NAMES[pred], name)
            shown = True

    cpu_after = psutil.cpu_percent(interval=None)

    return {
        "avg": np.mean(latencies),
        "std": np.std(latencies),
        "cpu": (cpu_before + cpu_after) / 2,
        "acc": correct / N_RUNS
    }

# ======================
def main():
    image_paths = []
    for cls in CLASS_NAMES:
        image_paths += glob(f"{DATASET_DIR}/{cls}/*.jpg")

    print("\n=== Inference Benchmark (RPi5 + lgpio) ===\n")

    results = {}

    for name, path in MODELS.items():
        print(f"Running {name}...")
        results[name] = benchmark_model(name, path, image_paths)

    print("\n+-------------+----------+----------+----------+----------+")
    print("| Model       | Avg(ms)  | Std(ms)  | CPU(%)   | Acc      |")
    print("+-------------+----------+----------+----------+----------+")
    for k, v in results.items():
        print(f"| {k:<11} | {v['avg']:>7.2f} | {v['std']:>7.2f} | {v['cpu']:>7.1f} | {v['acc']:.3f} |")
    print("+-------------+----------+----------+----------+----------+")

    # 清理 GPIO
    lgpio.gpiochip_close(h)

# ======================
if __name__ == "__main__":
    main()

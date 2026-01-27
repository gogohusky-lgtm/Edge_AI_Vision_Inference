import os
import time
import json
import argparse
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ====== 設定 ======
MODEL_PATH = "pet_classifier_int8_PTQ.tflite"
INPUT_SIZE = (160, 160)

# ====== class index ======
with open("class_indices.json", "r") as f:
    CLASS_INDICES = json.load(f)
IDX2CLASS = {v: k for k, v in CLASS_INDICES.items()}


def get_validation_images():
    with open("list.json", "r") as f:
        images = json.load(f)
    # Windows → Linux path 相容
    images = [p.replace("\\", os.sep) for p in images]
    return images


def load_image(path, interpreter):
    img = Image.open(path).convert("RGB")
    img = img.resize(INPUT_SIZE)
    img = np.array(img)

    input_details = interpreter.get_input_details()
    dtype = input_details[0]["dtype"]
    scale, zero_point = input_details[0]["quantization"]

    if dtype == np.uint8:
        # ===== INT8 / UINT8 =====
        img = img.astype(np.float32) / 255.0
        img = img / scale + zero_point
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        # ===== FP32 / FP16 =====
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)


def run_inference(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def benchmark(batch_size):
    images = get_validation_images()

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # ===== cold start =====
    img0 = load_image(images[0], interpreter)
    t0 = time.perf_counter()
    _ = run_inference(interpreter, img0)
    cold_ms = (time.perf_counter() - t0) * 1000

    # ===== steady state =====
    correct = 0
    start = time.perf_counter()

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        for path in batch:
            img = load_image(path, interpreter)
            out = run_inference(interpreter, img)
            pred = int(np.argmax(out))
            gt = os.path.basename(os.path.dirname(path))
            if IDX2CLASS[pred] == gt:
                correct += 1

    total_time = time.perf_counter() - start
    avg_ms = (total_time / len(images)) * 1000
    throughput = len(images) / total_time

    return {
        "device": f"RPi5: {MODEL_PATH}",
        "batch": batch_size,
        "cold_ms": cold_ms,
        "avg_ms": avg_ms,
        "throughput": throughput,
        "fps": throughput,
        "accuracy": correct / len(images),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    r = benchmark(args.batch)

    print("\n=== RPi5 Benchmark ===")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

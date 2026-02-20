import os
import time
import json
import argparse
import csv
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

MODEL_PATH = "pet_classifier_fp32.tflite"
model_base = os.path.splitext(os.path.basename(MODEL_PATH))[0]
file_name = f"{model_base}_benchmark.csv"
INPUT_SIZE = (160, 160)
STEADY_RUNS = 30

# ---------- Load class indices ----------
with open("class_indices.json", "r") as f:
    CLASS_INDICES = json.load(f)
IDX2CLASS = {v: k for k, v in CLASS_INDICES.items()}

# ---------- Load image list ----------
def get_image_list():
    with open("list.json", "r") as f:
        images = json.load(f)
    images = [p.replace("\\", os.sep) for p in images]
    return images

# ---------- Image preprocessing ----------
def load_image(path, interpreter):
    img = Image.open(path).convert("RGB")
    img = img.resize(INPUT_SIZE)
    img = np.array(img)

    input_details = interpreter.get_input_details()
    dtype = input_details[0]["dtype"]
    scale, zero_point = input_details[0]["quantization"]

    if dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
        img = img / scale + zero_point
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

def run_inference(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])

# ---------- CSV Init ----------
def init_csv():
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "latency_ms"])

# ---------- Reboot / Process Cold ----------
def run_cold(mode):
    images = get_image_list()
    img_path = images[0]

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    img = load_image(img_path, interpreter)

    t0 = time.perf_counter()
    _ = run_inference(interpreter, img)
    latency = (time.perf_counter() - t0) * 1000

    print(f"{mode} latency: {latency:.2f} ms")

    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([mode, latency])

# ---------- Steady ----------
def run_steady():
    images = get_image_list()
    img_path = images[0]

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    img = load_image(img_path, interpreter)

    # warm-up
    _ = run_inference(interpreter, img)

    latencies = []

    for _ in range(STEADY_RUNS):
        t0 = time.perf_counter()
        _ = run_inference(interpreter, img)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

    print("\n=== STEADY STATE ===")
    print(f"Mean: {np.mean(latencies):.2f} ms")
    print(f"Median: {np.median(latencies):.2f} ms")
    print(f"Std: {np.std(latencies):.2f} ms")
    print(f"Min: {np.min(latencies):.2f} ms")
    print(f"Max: {np.max(latencies):.2f} ms")

    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        for l in latencies:
            writer.writerow(["steady", l])

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["reboot_cold", "process_cold", "steady"]
    )
    args = parser.parse_args()

    init_csv()

    if args.mode in ["reboot_cold", "process_cold"]:
        run_cold(args.mode)
    elif args.mode == "steady":
        run_steady()

import os
import time
import json
import argparse
import csv
import numpy as np
from PIL import Image
import trt_infer   # TensorRT wrapper

ENGINE_PATH = "pet_classifier_fp32.engine"   # TensorRT engine
model_base = os.path.splitext(os.path.basename(ENGINE_PATH))[0]
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
def load_image(path):
    """
    使用 jetson.update.py 的優化方式：
    - numpy float32
    - 正規化到 [0,1]
    - CHW 格式
    - copy() 確保記憶體連續
    - expand_dims 增加 batch 維度
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1).copy()
    arr = np.expand_dims(arr, axis=0)
    return arr

def run_inference(infer_engine, img_np):
    out_raw = infer_engine.infer(img_np)   # 保持 numpy，不轉 list
    return np.array(out_raw, dtype=np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

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

    infer_engine = trt_infer.TrtInfer(ENGINE_PATH)
    img_np = load_image(img_path)

    t0 = time.perf_counter()
    out = run_inference(infer_engine, img_np)
    latency = (time.perf_counter() - t0) * 1000

    probs = softmax(out)
    pred = int(np.argmax(probs))

    print(f"{mode} latency: {latency:.2f} ms")
    # print(f"Pred class: {IDX2CLASS[pred]}")

    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([mode, latency])

# ---------- Steady ----------
def run_steady():
    images = get_image_list()
    img_path = images[0]

    infer_engine = trt_infer.TrtInfer(ENGINE_PATH)
    img_np = load_image(img_path)

    # warm-up
    _ = run_inference(infer_engine, img_np)

    latencies = []

    for _ in range(STEADY_RUNS):
        t0 = time.perf_counter()
        out = run_inference(infer_engine, img_np)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

    # probs = softmax(out)
    # pred = int(np.argmax(probs))

    print("\n=== STEADY STATE ===")
    print(f"Mean: {np.mean(latencies):.2f} ms")
    print(f"Median: {np.median(latencies):.2f} ms")
    print(f"Std: {np.std(latencies):.2f} ms")
    print(f"Min: {np.min(latencies):.2f} ms")
    print(f"Max: {np.max(latencies):.2f} ms")
    # print(f"Pred class: {IDX2CLASS[pred]}")

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

"""
Note:
This script is used for inference comparison only.
The underlying TensorRT engine was built and validated separately.
C++ bindings and build scripts are intentionally excluded
to keep this repository focused on system-level comparison.
"""

import os
import time
import json
import argparse
import numpy as np
from PIL import Image
import trt_infer

INPUT_SIZE = (160, 160)
VAL_DIR = "validation_backup"

with open("class_indices.json", "r") as f:
    CLASS_INDICES = json.load(f)
IDX2CLASS = {v: k for k, v in CLASS_INDICES.items()}


def load_batch(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize(INPUT_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # CHW
        imgs.append(arr)
    return np.stack(imgs, axis=0)


def get_validation_images():
    images = []
    with open ("list.json", "r") as f:
        images = json.load(f)
    images = [p.replace("\\", os.sep) for p in images]
    return images


def benchmark(engine_path, batch_size):
    images = get_validation_images()
    infer = trt_infer.TrtInfer(engine_path)

    # ===== cold start =====
    imgs = load_batch(images[:batch_size])
    t0 = time.time()
    _ = infer.infer(imgs.flatten().tolist())
    cold_ms = (time.time() - t0) * 1000

    # ===== steady =====
    correct = 0
    start = time.time()

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        imgs = load_batch(batch)
        outputs = infer.infer(imgs.flatten().tolist())
        outputs = np.reshape(outputs, (-1, len(CLASS_INDICES)))
        preds = np.argmax(outputs, axis=1)

        for p, path in zip(preds, batch):
            gt = os.path.basename(os.path.dirname(path))
            if IDX2CLASS[int(p)] == gt:
                correct += 1

    total = time.time() - start
    avg_ms = (total / len(images)) * 1000
    throughput = len(images) / total

    return {
        "device": "Jetson Nano 2GB",
        "engine": os.path.basename(engine_path),
        "batch": batch_size,
        "cold_ms": cold_ms,
        "avg_ms": avg_ms,
        "throughput": throughput,
        "fps": throughput,
        "accuracy": correct / len(images)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", help="fp32.trt or fp16.trt")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    r = benchmark(args.engine, args.batch)

    print("\n=== Jetson Nano Benchmark ===")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

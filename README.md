# Raspberry Pi 5 上的邊緣 AI 視覺推論
### （含工程層面與 Jetson Nano 的比較）

**核心重點：** 端到端邊緣推論、系統取捨、硬體感知部署

---

## 專案概述

此專案展示了一個 **完整的端到端邊緣 AI pipeline**：

- PC 端模型訓練與量化（FP32 / FP16 / INT8 PTQ / INT8 QAT）
- **Raspberry Pi 5（CPU，TFLite + XNNPACK）** 上的裝置端推論
- **GPIO LED 硬體回饋**，用於感知到動作的驗證
- 與 **Jetson Nano 2GB（TensorRT FP32 / FP16）** 的工程比較

> 目標並非 **追逐 benchmark 數字**，而是理解 **硬體限制如何影響實際的邊緣 AI 部署**。

---
## 此專案解決了什麼問題？
此專案不只停留在「模型能跑」或「FPS 看起來不錯」，希望進一步回答：
- 哪個平台真正適合 **事件驅動、單張影像推論**？
- **GPU 加速**在低記憶體邊緣裝置上是否總是有幫助？
- 不同的 **量化策略在真實硬體上的表現**如何？

---

## 主要工程發現

### Raspberry Pi 5（4GB，CPU 推論）
- 在 batch = 1 與 batch > 1 下延遲穩定
- 記憶體使用可預測
- 非常適合 **事件驅動、低延遲的邊緣應用**

### Jetson Nano（2GB，TensorRT）
- GPU 加速可行，但：
  - 執行期記憶體限制導致 batch > 1 無法運行
  - GPU 上下文初始化與 CPU–GPU 記憶體傳輸主導延遲
- 對單張影像推論來說 **不一定更快**

**系統層級結論**

> 在低延遲、事件驅動的邊緣 AI 情境中，  
> **Raspberry Pi 5 的 CPU 推論可能優於 Jetson Nano 的 GPU 推論**  
> 當考量端到端系統行為時。

---

## Demo（Raspberry Pi 5）
**邊緣推論 + GPIO LED 回饋**  

感知 → 推論 → 硬體動作

示範影片：  
https://youtube.com/shorts/biKfEp-H_zw

系統架構：

![系統架構圖](docs/system_arch.png)

---

**技術細節、基準測試與實作紀錄如下**

---
---

## 1. 系統架構

系統設計為一個簡潔但完整的 **感知到動作邊緣管線**：

- PC 上的離線模型訓練與最佳化
- Raspberry Pi 5 上的裝置端推論
- 基於 GPIO 的硬體回饋，用於驗證即時整合

---

## 2. 模型訓練與最佳化（PC 端）

### 資料集
- Oxford-IIIT Pet Dataset
- 使用官方 XML 標註進行 ROI 裁切
- 類別：
  - 貓
  - 狗
  - 其他

### 基礎模型
- MobileNetV2（ImageNet 預訓練）
- 輸入大小：160 × 160

### 最佳化策略

| 模型類型 | 方法 |
|-----------|--------|
| FP32 | 基準 |
| FP16 | 訓練後 float16 轉換 |
| INT8 PTQ | 訓練後量化 |
| INT8 QAT | 量化感知訓練（僅 CPU） |

FP32 特意保留作為參考，用於評估 **真實世界的取捨**，而非僅僅理論上的提升。

---

## 3. Raspberry Pi 5 邊緣推論

### 推論框架
- `tflite-runtime`
- 後端：XNNPACK（CPU）

### 評估指標
- 平均延遲（毫秒）
- 延遲標準差
- CPU 使用率
- 分類準確率

### 結果摘要

| 模型 | 平均延遲 (ms) | CPU % | 準確率 |
|------|------------------|-------|----------|
| FP32 | 11.76 | 7.9 | 1.000 |
| FP16 | 12.77 | 13.6 | 1.000 |
| INT8 PTQ | 22.53 | 6.0 | 1.000 |
| INT8 QAT | 21.34 | 7.8 | 0.950 |

---

## 4. GPIO 硬體回饋

推論結果映射至 LED：

| 類別 | GPIO 腳位 | LED |
|------|---------|-----|
| 貓 | GPIO 16 | 紅色 |
| 狗 | GPIO 20 | 黃色 |
| 其他 | GPIO 21 | 綠色 |

每次推論僅啟動 **一個 LED**，驗證：
- 推論結果的即時整合
- 硬體回應的確定性

> 專案刻意避推論決定後之機構動作，以保持焦點在 **邊緣 AI 系統行為**，而非硬體複雜度。

---

## 5. RPi 5 與 Jetson Nano 工程比較

### 範疇
- 任務：單張影像寵物分類
- 情境：事件驅動推論
- 重點：
  - 端到端延遲
  - 記憶體限制
  - 系統穩定性

### 測試環境

| 裝置 | 記憶體 | 框架 | 後端 |
|------|--------|----------|--------|
| Raspberry Pi 5 | 4GB | TFLite | XNNPACK（CPU） |
| Jetson Nano | 2GB | TensorRT | CUDA |

### 效能摘要

#### Raspberry Pi 5
- 延遲約 ~12 ms（FP32 / FP16）
- 在 batch > 1 下行為穩定
- CPU 為主但可預測

#### Jetson Nano 2GB
- 延遲約 ~43–45 ms（batch = 1）
- batch > 1 因執行期記憶體分配失敗 (std::bad_alloc)
- 單張影像推論時 GPU 的 overhead 佔主導

詳細數據置於 benchmarks 夾中

---

## 6. 工程結論

- 量化效益 **依平台而異**
- GPU 加速在少記憶體的邊緣裝置上 **並非絕對優勢**
- 系統層級的評估對部署決策非常重要

此專案強調 **工程判斷，而非 benchmark 數字**。

---

## 7. 檔案目錄結構
```text
Edge_AI_model_optimization/
├── benchmarks/
│   ├── rpi5_full_results.csv
│   ├── jetson_nano_full_results.csv
│   └── methodology.md
│
├── docs/
│   ├── wiring.md
│   ├── requirements.txt
│   ├── system_arch.png
│   └── system_architecture.png
│
├── edge_inference/
│   ├── inference.py             # RPi5 推論與效能
│   ├── RPi5_inference.py        # 推論比較用(RPi5端)
│   ├── Jetson_inference.py      # 推論比較用(Jetson端)
│   └── class_indices.json       # 推論分類之標示
│
├── models/
│   ├── pet_classifier_fp32.tflite
│   ├── pet_classifier_fp16.tflite
│   ├── pet_classifier_int8_PTQ.tflite
│   ├── pet_classifier_int8_QAT_CPU.tflite
│   ├── fp16.trt
│   └── fp32.trt
│
├── pc_pipeline/
│   ├── Cropping_Group.py        # 依 XML 標註進行 ROI 裁切
│   ├── resize.py                # 影像調整為 160x160
│   ├── Training.py              # FP32 / FP16 / PTQ 模型訓練
│   └── QAT_CPU.py               # CPU 上進行 INT8 QAT
│
├── screenshots_demo
│   ├── Comp_RPi5_summary.png    # 推論比較 summary 截圖 (RPi5端)
│   ├── Comp_Jetson_summary.png  # 推論比較 summary 截圖 (Jetson端)
│   └── inference_summary.png    # 推論 summary 截圖
│
├── validation_backup/           # 推論用驗證照片
│
└── README.md

```
---
## License

The source code in this repository is released under the MIT License.

Demo materials, including videos, photos, logs, and generated data under the following directories are provided for demonstration purposes only and are NOT covered by the MIT License:

screenshots_demo/

These materials may not be redistributed or reused without explicit permission.
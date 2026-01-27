# **Edge AI model optimization and deployment on Raspberry Pi 5**
## Executive Summary

This project demonstrates an end-to-end edge AI workflow, from model training and quantization on PC to real-time inference deployment on Raspberry Pi 5. Multiple TensorFlow Lite models (FP32, FP16, INT8 PTQ, INT8 QAT) are systematically compared in terms of latency, CPU usage, and inference accuracy.

The project emphasizes practical engineering trade-offs in edge deployment rather than pursuing maximum model accuracy. A lightweight GPIO-based LED output is used to visualize inference decisions directly on-device.

This repository serves as a practical workflow example for model optimization and on-device inference on Raspberry Pi 5, with real-time GPIO feedback, tailored for embedded and edge AI engineering roles.


## Raspberry Pi 5 上的 Edge AI 模型最佳化與部署 (案例研究：貓 / 狗 影像分類)

本專案展示一個完整的 Edge AI 工作流程，重點放在 模型最佳化、部署，以及在 Raspberry Pi 5 上的效能比較。

與其打造一個完整的消費性產品，本專案刻意將主軸放在：

- 原始資料集的前處理流程
- 模型訓練與不同量化策略之四種模型格式：
    - FP32
    - FP16
    - INT8 (Post-Training Quantization, PTQ)
    - INT8 (Quantization-Aware Training, QAT)
- 在資源受限的邊緣裝置上進行推論
- 比較並量測：
    - 推論延遲(latency)
    - CPU 使用率
    - 分類準確率
- 驗證 `tflite-runtime` 在邊緣裝置上的實務限制
- 整合 GPIO 輸出，以 LED 顯示推論結果

貓 / 狗分類僅作為案例情境，用來探討實務中 Edge AI 系統的取捨與限制。

## 系統架構
![系統架構圖](docs/system_architecture.png)

## Demo video (Edge Inference on Raspberry Pi)
本影片展示在 Raspberry Pi 5 上進行即時推論，並透過 GPIO 輸出回饋。影片暫以 shorts 方式呈現。

https://youtube.com/shorts/biKfEp-H_zw

## 模型訓練（PC 端）
### 資料集
- Oxford-IIIT Pet Dataset
- 使用官方提供的 XML 標註進行辨視用區域 ROI 裁切
- 分類類別：
    - cats
    - dogs
    - others
### 基礎模型
- MobileNetV2（ImageNet 預訓練權重）
- 輸入尺寸：160 × 160

### 模型最佳化策略
本專案刻意同時保留 FP32 作為基準模型，以量化不同模型最佳化策略在邊緣裝置上的實際影響。

| 模型類型     | 方法             |
| -------- | -------------- |
| FP32     | 基準模型           |
| FP16     | 訓練後 float16 轉換 |
| INT8 PTQ | 使用代表資料集進行量化    |
| INT8 QAT | 量化感知訓練（CPU）    |

## 邊緣裝置推論（Raspberry Pi 5）
### 推論框架
`tflite-runtime`
### 比較指標
- 平均推論延遲（毫秒）
- 延遲標準差
- CPU 使用率
- 分類準確率

所有模型皆使用相同推論流程，以確保比較公平性。

### 比較輸出

| Model | Avg Latency (ms) | CPU % | Accuracy |
| ----- | ---------------- | ----- | -------- |
| FP32  | 11.76            | 7.9   | 1.000    |
| FP16  | 12.77            | 13.6  | 1.000    |
| PTQ   | 22.53            | 6.0   | 1.000    |
| QAT   | 21.34            | 7.8   | 0.950    |


### GPIO 輸出（LED 對應）
推論結果以三顆 LED 顯示：
| 分類 | GPIO 腳位 | LED |
|------|-------|-----|
| cats   | GPIO 16 | 紅色 |
| dogs   | GPIO 20 | 黃色 |
| others | GPIO 21 | 綠色 |

每次推論僅點亮對應類別的一顆 LED。
```
說明：
本專案使用 LED 作為最小化的硬體回饋介面，用以驗證：
- 邊緣推論結果與實體 GPIO 控制的整合流程
- 推論決策與硬體輸出之即時性

實際的餵食機構或機械致動元件刻意未納入設計範圍，以避免硬體設計掩蓋 Edge AI 模型效能比較的主軸。
```
## 測試與開發硬體環境（Reference）
### 模型訓練用 PC:
- Laptop: MSI GP62 2QE
- GPU: NVIDIA (2GB VRAM)
- OS: Windows

### 邊緣裝置:
- Raspberry Pi 5
- OS: Raspberry Pi OS (64-bit)

## 軟體元件
### PC 端
- Python >= 3.8
- TensorFlow
- TensorFlow Model Optimization Toolkit
- OpenCV
- NumPy
- Matplotlib
### Raspberry Pi 5 端
- Python >= 3.9
- tflite-runtime
- lgpio
- NumPy
- OpenCV
- psutil
- Matplotlib

Raspberry Pi 上 不需要安裝 TensorFlow。

PC 與 Raspberry Pi 5 端的軟體環境刻意分離，以符合實務中「模型訓練與邊緣部署」的典型工作流程。

補充說明：
- 模型訓練主要使用 TensorFlow GPU 版本
- CUDA / cuDNN 版本依使用者系統環境而異，未強制綁定
- 若 GPU 記憶體不足，QAT 可改以 CPU-only 模式執行（本專案已實際驗證）

## 啟動順序
1. 對 Oxford-IIIT Pet Dataset 進行辨視用大頭照裁切 [Cropping_Group.py]
2. 縮小大頭照/對照用之其他室內照片至統一尺寸 160x160 [resize.py]
3. 分類訓練及產生模型檔 (FP32/FP16/INT8 PTQ) [Training.py]
4. 產生模型檔 INT8 QAT [QAT_CPU.py]
5. 執行推論 [inference.py]

預期之結果：
- 資料集的前處理產出裁切影像（160x160）
- 模型訓練（含多種量化策略）產出 .tflite 模型
- Raspberry Pi 上部署及推論產出 GPIO LED 回饋

## 系統行為

- 推論完全於 Raspberry Pi 5 上進行 (非雲端推論)
- 推論用資料完全於 Raspberry Pi 5 上處理
- 分類結果以 GPIO 輸出對應
- LED 狀態反應即時推論結果

本設計專注於 low-latency 邊緣推論而非用戶端之 UI。這種最小化的硬體回饋機制，能在不引入額外 I/O 或機械變數的情況下，評估推論效能與系統延遲。


## 觀察與結論
- FP16 在 CPU 使用率上通常優於 FP32，且準確率相近
- INT8 PTQ 可能因量化 / 反量化成本而增加延遲
- INT8 QAT 可提升量化模型穩定性，但在 CPU 上不一定優於 FP16
- 模型最佳化效果與實際硬體平台高度相關

以上結果顯示，模型最佳化策略的效益高度依賴實際部署平台，無法僅依理論或單一指標做判斷。

## 設計決策 & 已知限制
### 設計取捨說明
- 未實作實際餵食機構
- 未使用實體感測器作為推論觸發
- 專注於**模型效能與邊緣部署限制**的工程層面
- 此取捨使專案更適合用來展示工程能力，而非產品雛形。

### 未來延伸（選擇性）
- 加入感測器或按鈕作為推論觸發
- 整合相機進行即時影像推論
- 匯出效能比較結果為 CSV
- 與 GPU / NPU 加速器進行比較

----
----
## RPi5 vs Jetson Nano 推論比較

本節提供一個工程導向的比較，針對 **Raspberry Pi 5 (4GB)** 與 **Jetson Nano (2GB)** 在單張影像寵物分類任務上的表現。 此比較的目標並非追求最高基準分數， 而是評估在真實部署限制下的邊緣推論行為。

### 比較範疇 
- 任務：單張影像狗/貓分類 
- 輸入：160×160 RGB 影像 
- 情境：事件觸發推論 
- 重點： 
    - 延遲一致性 
    - 系統穩定性 
    - 記憶體限制 
    - 可重現性

由於 Jetson Nano 2GB 型號在執行時的記憶體限制， 批次推論 (>1) 被刻意排除。 

---
### 測試環境 

| 裝置 | 記憶體 | 框架 | 後端 | 
|------|--------|----------|---------| 
| Raspberry Pi 5 | 4GB | TFLite | XNNPACK (CPU) | 
| Jetson Nano | 2GB | TensorRT | CUDA | 

### Models

Jetson Nano 的 TensorRT 引擎是離線生成的， 並提供以確保結果的可重現性。 在裝置上重新建構引擎可能需要額外的記憶體， 且不包含在此資料庫中。

--- 
### 效能摘要 
#### Raspberry Pi 5 (TFLite / XNNPACK) 

| 精度 | 批次 | 平均延遲 (ms) | FPS | 準確率 | 
|-------|------|--------|------|------| 
| FP32 | 1 | ~12.4 | ~80.9 | 1.000 | 
| FP32 | 100 | ~12.4 | ~80.5 | 1.000 | 
| FP16 | 1 | ~12.4 | ~80.7 | 1.000 | 
| FP16 | 100 | ~12.4 | ~80.8 | 1.000 | 
| INT8 (PTQ) | 1 | ~21.4 | ~46.6 | 0.9969 | 
| INT8 (QAT) | 1 | ~21.3 | ~47.0 | 0.9907 | 

**觀察** 批次大小對延遲或吞吐量影響極小， 顯示推論行為受限於 CPU，且記憶體使用穩定。 

--- 
#### Jetson Nano 2GB (TensorRT) 
| 精度 | 批次 | 平均延遲 (ms) | FPS | 準確率 | 
|-----|------|---------|----|-----| 
| FP32 | 1 | ~45.5 | ~22.0 | 1.000 | 
| FP16 | 1 | ~43.1 | ~23.2 | 1.000 | 

**觀察** Jetson Nano 可達成 GPU 加速推論， 但 2GB 型號的可用記憶體限制了批次推論的實用性。 批次推論 (>1) 導致執行時配置失敗 (`std::bad_alloc`)，因此未納入最終測量。 

--- 
### 工程結論 
- **RPi5** 提供高度穩定且可重現的推論效能， 適合需要緊密整合的多任務邊緣系統。 
- **Jetson Nano** 擅長 GPU 加速的單張影像推論， 但 2GB 型號的記憶體限制大幅限制了批次策略。 
- RPi5 的 INT8 量化在準確率與推論效能間取得平衡， 是長時間邊緣部署的可行選項。 - 在產品導向的展示中， 系統穩定性與可解釋性優先於峰值吞吐量。 
--- 
### ✅ 結論 
雖然 Jetson Nano 提供較低延遲的 GPU 推論， Raspberry Pi 5 在受限記憶體環境下展現更佳的系統穩健性與可擴展性。 對於即時、事件驅動的邊緣 AI 應用， RPi5 提供更可預測且可重現的部署平台。

## 檔案目錄結構
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

## License Notice

The source code in this repository is released under the MIT License.

Demo materials, including videos, photos, logs, and generated data under the following directories are provided for demonstration purposes only and are NOT covered by the MIT License:

- screenshots_demo/

These materials may not be redistributed or reused without explicit permission.

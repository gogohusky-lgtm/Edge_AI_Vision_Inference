# 邊緣 AI 視覺推論最佳化與部署 (Raspberry Pi 5 / Jetson Nano - 工程案例研究)
**核心重點：** 在受限硬體上理解模型 × Runtime × 記憶體行為如何共同影響真實推論效能。
---
## 專案概述
本專案展示一個端到端的邊緣 AI 視覺推論展示，從模型訓練與量化，到在實際嵌入式裝置上的部署與效能分析。說明了**模型最佳化、runtime 狀態，以及系統記憶體限制**如何共同決定邊緣裝置上的真實推論效能。

以貓 / 狗影像分類任務作為參考工作負載，推論在以下平台上實作並評估：

- Raspberry Pi 5 (4GB) 使用 TensorFlow Lite（CPU, XNNPACK）
- Jetson Nano (2GB) 使用 TensorRT（GPU）

>本專案的核心價值不在於追求最高 FPS，而在於理解在資源受限硬體上，推論效能如何受到模型、runtime 與系統狀態共同影響，並據此做出可部署的設計選擇。

## 主要工程發現：
- Raspberry Pi 5 提供穩定且可預測的推論延遲，適合事件驅動的邊緣應用。
- Jetson Nano 提供 GPU 加速，但在 2GB 型號上，冷啟動、runtime 狀態與記憶體可用性主導效能。
- GPU 推論速度不僅取決於模型精度，還受到 **runtime 初始化與作業系統層級行為**影響。

## 結果快照（Batch = 1, Event-Driven Inference）

| 平台 | 冷啟動 | 穩態延遲 | 關鍵特性 |
|------|--------|----------|----------|
| RPi5 (CPU) | ~15 ms | ~12 ms | 穩定、可預測 |
| Jetson Nano (GPU) | 4–7 s | 15–25 ms | Runtime & 記憶體主導 |

> 關鍵發現： 對於單影像、事件驅動的推論，在 2GB GPU 裝置上，系統狀態與 CUDA 初始化成本往往比模型精度更影響實際延遲。

## Demo（Raspberry Pi 5）
展示從感知 → 推論 → 硬體動作的完整閉環，模擬實際邊緣設備中的即時控制流程。

示範影片：https://youtube.com/shorts/biKfEp-H_zw


本專案另有容器化的展示，可參考 https://github.com/gogohusky-lgtm/Dockerized_Edge_AI_Vision_Inference

---
技術細節、benchmark 測試與實作紀錄如下

## 系統概覽
- 任務：貓 / 狗影像分類
- 輸入大小：160×160 RGB
- 推論模式：事件驅動（batch = 1）
- 研究重點：延遲行為、記憶體限制、冷啟動

![系統架構圖](docs/system_arch.png)

---
## 工程比較與洞察
### 1. Raspberry Pi 5 — TensorFlow Lite (XNNPACK)
代表性結果（Batch = 1）

|精度	|冷啟延遲 (ms)	|平均延遲 (ms)	|準確率|
|--------|----------|------------|-------------|
|FP32|	~14.6|	~12.4|	1.000|
|FP16|	~15.1|	~12.4|	1.000|
|INT8 PTQ|	~20.9|	~21.4|	0.9969|
|INT8 QAT|	~33.6|	~21.3|	0.9907|

觀察
- 延遲與記憶體使用高度穩定
- 量化在 CPU 上不保證加速
- 適合長時間運行的邊緣服務

### 2. Jetson Nano (2GB) — TensorRT (GPU)
#### 記憶體最佳化前

| 精度	| 冷啟延遲 (ms) |	平均延遲 (ms) |
|--------|----------|------------|
|FP16|	~5,596|	~43.1|
|FP32|	~10,976|	~45.5|


#### 記憶體最佳化後
| 精度	| 冷啟延遲 (ms)	| 平均延遲 (ms)|
|--------|----------|------------|
|FP16|	~1,411–6,019 |	~15–26|
|FP32|	~1,443–7,215 |	~16–32|

Runtime 關鍵洞察
- 記憶體與 runtime 調整後，Jetson Nano 的平均延遲從 ~45ms 改善至 ~15–25ms 區間，但冷啟延遲仍受 CUDA 初始化主導。
- 冷啟延遲主要由 TensorRT engine 載入與 CUDA context 初始化主導
- Workspace size 影響 engine 行為
- 在記憶體壓力下，平均延遲可能波動，FP16 不一定優於 FP32
- 執行順序與 runtime 狀態顯著影響延遲
- 在 2GB RAM 限制下，使用 batch=1 可作為穩定部署選擇

## 工程結論
- 在低記憶體 GPU 裝置上，系統行為往往比模型精度更主導實際延遲。即，GPU 推論 ≠ 一定更快
- 冷啟延遲是邊緣部署中的重要指標
- 記憶體限制會改變最佳設計選擇
- 系統可預測性往往比峰值吞吐量更重要

---

## 方法論與完整基準測試
完整結果：
- `benchmarks/edge_inference_benchmark.csv`
- `benchmarks/methodology.md`

## GPIO 硬體回饋 (RPi5)
推論結果映射至 LED：

|類別|	GPIO 腳位|	LED|
|---|-----|---|
|貓|	GPIO 16|	紅色|
|狗|	GPIO 20|	黃色|
|其他|	GPIO 21|	綠色|

---

## 檔案目錄結構

```text
Edge_AI_model_optimization/
├── benchmarks/
│   ├── edge_inference.csv
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
│   ├── fp16.trt                 # 優化後FP16 engine
│   └── fp32.trt                 # 優化後FP32 engine
│
├── pc_pipeline/
│   ├── Cropping_Group.py        # 依 XML 標註進行 ROI 裁切
│   ├── resize.py                # 影像調整為 160x160
│   ├── Training.py              # FP32 / FP16 / PTQ 模型訓練
│   └── QAT_CPU.py               # CPU 上進行 INT8 QAT
│
├── screenshots_demo
│   ├── Comp_RPi5_summary.png        # 推論比較 summary 截圖 (RPi5端)
│   ├── Comp_Jetson_bfr_summary.png  # 優化前推論比較 summary 截圖 (Jetson端)
│   ├── Jetson_run-1.txt             # 優化後推論比較綜整 (Jetson端,CLI[1]~[4],Desktop[5], FP16先跑)
│   ├── Jetson_run-2.txt             # 優化後推論比較綜整 (Jetson端,CLI[1]~[4],Desktop[5], FP32先跑)
│   ├── Comp_Jetson_aft_summary-1.png  # 優化後推論比較截圖 (Jetson端，Desktop, FP16先跑)
│   ├── Comp_Jetson_aft_summary-2.png  # 優化後推論比較截圖 (Jetson端，Desktop, FP32先跑)
│   └── RPi5_inference_GPIO.png        # RPi5 (GPIO) 推論 summary 截圖
│
├── validation_backup/           # 推論用驗證照片
│
└── README.md
```

## License

The source code in this repository is released under the MIT License.

Demo materials, including videos, photos, logs, and generated data under the following directories are provided for demonstration purposes only and are NOT covered by the MIT License:

screenshots_demo/

These materials may not be redistributed or reused without explicit permission.

---
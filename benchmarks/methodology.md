# Benchmark 測試方法論

本文說明本專案用於比較不同邊緣裝置推論效能的測試設計與控制條件。

本測試目的並非追求理論峰值效能，而是模擬實際邊緣部署情境下的可重現推論行為。

---

## 1. 任務定義

- 任務：單張影像寵物分類（貓 / 狗 / 其他）
- 輸入尺寸：160 × 160 RGB
- 模型架構：MobileNetV2-based classifier
- 輸出：Softmax 機率向量

所有平台使用相同模型權重與相同驗證影像。

---

## 2. 推論情境設計

測試模擬「事件觸發型」邊緣應用：

- 單張影像觸發推論
- 無離線批次排程
- 無非同步 pipeline
- 無多執行緒優化
- 無預先 warm-up

目標是觀察單次請求下的延遲與穩定性。

---

## 3. 批次定義

Batch size 定義為一次呼叫推論 API 時輸入影像數量。

- Raspberry Pi 5 (CPU)：測試 batch = 1 與 batch = 100
- Jetson Nano (2GB)：僅測試 batch = 1

原因：

在 Jetson Nano 2GB 上，batch > 1 會因 runtime 記憶體不足導致配置失敗（std::bad_alloc），因此最終設計選擇為 batch = 1。

---

## 4. 時間測量定義

### Cold Start Latency

包含：

- Engine / model 載入
- CUDA context 初始化（Jetson）
- TensorRT runtime 建立
- 第一次推論執行

Cold start 定義為「首次完整推論完成所需時間」。

---

### Steady-State Latency

- 連續執行 10 次推論
- 計算平均值 (avg latency)
- 同時計算最小值 (min latency)

此設計用於觀察 runtime 穩定後的延遲行為。

---

## 5. 記憶體與系統狀態控制

Jetson Nano 測試時：

- 推論前執行 `free -m` 紀錄可用記憶體
- 比較 Desktop 與 CLI 模式
- 進行 reboot 對照實驗
- 測試 FP16 → FP32 與 FP32 → FP16 執行順序差異

目的是觀察：

- 記憶體壓力對 latency 影響
- CUDA context 是否殘留
- runtime stateful 行為

---

## 6. 準確率評估

- 使用固定 validation subset
- 所有模型與裝置使用相同影像
- 不進行任何裝置特定後處理

準確率僅用於確保推論結果一致性，本研究重點為延遲與部署行為。

---

## 7. 限制說明

- 未測試長時間連續運行（>1 小時）
- 未測試多執行緒推論
- 未調整 Jetson power mode
- 未進行 TensorRT layer-level profiling

本研究重點為「系統層級行為觀察」，而非極限優化。

---

## 8. 設計哲學

本 benchmark 著重於：

- 可重現性
- 實際部署行為
- 記憶體限制下的設計選擇
- 冷啟動成本

而非僅報告單一最佳數字。

# 水產養殖 AI 監測系統 (Aquaculture AI Monitoring)

本專案整合 MQTT 資料蒐集、模型訓練、即時推論以及圖形化控制台，協助養殖場快速建立水質異常偵測流程。系統流程如下：

1. `mqtt_logger.py` 透過 MQTT 取得感測資料並寫入 `aquaculture_stream.csv`。
2. `train_classifier.py` 依門檻自動標籤資料、進行 K-fold 訓練並輸出 `aq_dnn.keras` 與 `aq_meta.pkl`，同時產生分析圖 `analysis_report.png`。
3. `live_infer.py` 載入模型與標準化器，對即時資料推論並回傳至 MQTT 告警主題。
4. `dashboard.py` 提供中英雙語 UI，可啟動/停止各項服務、檢視感測值與 AI 推論狀態。

---

## 目錄結構 (Project Layout)

- `mqtt_logger.py`：MQTT 資料蒐集器，將 pH / 溶氧 / 水溫存成 CSV。
- `train_classifier.py`：自動標籤 + K-fold 訓練 + 分析圖 + 模型輸出。
- `live_infer.py`：即時推論與 MQTT 告警發布。
- `dashboard.py`：Tkinter 控制台，整合啟動/停止、日誌視窗、即時數據顯示。
- `aquaculture_stream.csv`：蒐集到的水質資料（時間戳、ISO 時間、pH、溶氧、溫度）。
- `aq_dnn.keras` / `aq_meta.pkl`：訓練完成後的模型與標準化器/類別資訊。
- `analysis_report.png`：訓練完成後輸出的標籤分布與混淆矩陣圖。

---

## 系統需求 (Requirements)

- Python 3.9（與 TensorFlow for Windows 相容）
- 建議使用虛擬環境 (`python -m venv .venv`)
- 主要套件：
  ```bash
  pip install -r requirements.txt  # 若未建立可手動安裝：
  pip install paho-mqtt pandas numpy scikit-learn tensorflow joblib matplotlib
  ```

---

## 快速開始 (Quick Start)

1. **建立並啟動虛擬環境**
   ```powershell
   cd C:\SourceCode\5G
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt  # 或依需求安裝套件
   ```

2. **蒐集資料 (Data Logging)**
   ```bash
   python mqtt_logger.py
   ```
   - 預設訂閱 `jetsion/30AEA4903C20/WaterPH` / `WaterO2` / `WaterTemp`。
   - 若 CSV 為新檔，會自動建立標頭 `ts, iso_time, ph, o2, temp`。

3. **訓練模型 (Train Model)**
   ```bash
   python train_classifier.py --epochs 20 --splits 3
   ```
   - 預設門檻：`pH < 7.1`、`pH > 8.5`、`溶氧 < 7.64`、`溫度 > 26.3`。
   - 生成 `aq_dnn.keras`、`aq_meta.pkl`，並輸出 `analysis_report.png`（標籤分佈 + 混淆矩陣）、`training_curves.png`（連續的 Train/Val Accuracy、Loss 歷程）、`fold_comparison.png`（各折對照）。
   - 可透過參數調整門檻或摺數，例如 `--ph-low 6.8 --temp-high 27`。

4. **即時推論 (Live Inference)**
   ```bash
   python live_infer.py
   ```
   - 讀取最新模型，對感測值推論並發布至 `jetsion/30AEA4903C20/WaterQuality/Status`。
   - MQTT 消息格式：`<label>:<confidence>:<中文說明>`。

5. **圖形化控制台 (Dashboard)**
   ```bash
   python dashboard.py
   ```
   - 中英雙語介面顯示感測數據、門檻、AI 推論結果。
   - 可啟停資料蒐集/推論/訓練，日誌視窗同步顯示各程序輸出。

---

## 自動標籤邏輯 (Auto Labeling)

| 條件 Condition | 標籤 Label |
| -------------- | ---------- |
| `pH < 7.1`     | `pH_low`   |
| `pH > 8.5`     | `pH_high`  |
| `O2 < 7.64`    | `DO_low`   |
| `Temp > 26.3`  | `Temp_high`|
| 其他           | `normal`   |

- 門檻可於 `train_classifier.py` 參數調整，或在 CSV 中提供人工標籤。
- 若自動標籤後只有單一類別，訓練會直接拋出錯誤提醒需調整門檻或補資料。

---

## 分析圖 (analysis_report.png)

訓練腳本會輸出 `analysis_report.png`，內容包含：
- **左圖**：各類別樣本數（含中英文標籤）。
- **右圖**：召回率混淆矩陣，每格顯示「樣本數 / recall」。
- 若系統找不到支援中文字型，會自動 fallback 成英文顯示。

---

## 常見問題 (FAQ)

- **日誌中文字變問號**：腳本已自動設定 UTF-8；若仍遇到請確認終端/IDE 的輸出編碼。
- **MQTT Callback API 警告**：已採用 `callback_api_version=VERSION2`；若升級 paho-mqtt，再同步檢查簽名。
- **TensorFlow 警告**：`tf.function retracing` 與 `.h5` Legacy 提示可忽略，或視需求調整模型儲存格式。
- **分析圖中文字體缺失**：程式會自動偵測常見中文字型，若系統無對應字體則改用英文。

---

## Git 版本控制 (Git Workflow)

- `.venv/`、`__pycache__/`、`*.pyc` 已寫入 `.gitignore`，避免推送虛擬環境。
- 推送前建議確認 `.git/status` 乾淨，模型與 CSV 視情況可選擇不納入版控。

---

## 待辦與延伸 (Next Steps)

- 導入更多水質指標（氨氮、濁度…）需同步調整三支腳本與模型輸入。
- 依養殖情境自訂門檻或人工標註，提升模型準確度。
- 與排程/守護程式整合（Windows 服務、systemd、Docker）以便長期部署。
- 增加告警機制（Line Bot、Email、簡訊）或儀表板通知。

---

## License

若無特別聲明，預設採用專案擁有者的授權。請依實際需求更新 License。 
\n## 訓練可視化補充\n- `fold_comparison.png`：同時呈現三個 K-fold 的訓練/驗證準確率與損失曲線，方便並排比較（同圖不同顏色）。\n

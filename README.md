# GoldForecast-Hybrid System

**Tên đề tài:** Nghiên cứu mô hình lai SARIMAX–LSTM trong dự báo giá vàng quốc tế và mô phỏng chiến lược giao dịch ngắn hạn  
**English title:** A Hybrid SARIMAX–LSTM Approach for International Gold Price Forecasting and Short-Term Trading Simulation

GoldForecast-Hybrid System là hệ thống thực nghiệm phục vụ Đồ án 1, xây dựng quy trình hoàn chỉnh từ thu thập dữ liệu, xử lý dữ liệu, huấn luyện mô hình dự báo đến đánh giá thống kê và mô phỏng giao dịch ngắn hạn. Mô hình nghiên cứu chính là **SARIMAX–LSTM**, trong đó SARIMAX khai thác thành phần tuyến tính và biến ngoại sinh, còn LSTM học phần residual phi tuyến còn lại.

Hệ thống dự báo `target_log_return`, sau đó quy đổi về giá đóng cửa dự báo theo công thức:

```text
forecast_close[t] = close_lag_1[t] * exp(predicted_log_return[t])
```

Cách tiếp cận này giúp mô hình làm việc với chuỗi lợi suất ổn định hơn chuỗi giá gốc, đồng thời vẫn cho phép đánh giá kết quả theo giá vàng dự báo.

## 1. Đặc điểm chính

- Dữ liệu giá vàng quốc tế theo ngày từ Yahoo Finance, ticker `GC=F`.
- Biến ngoại sinh gồm USD Index, dầu thô, S&P 500, lợi suất trái phiếu Mỹ 10 năm, Fed Rate và CPI.
- Chia dữ liệu theo thời gian thành **Train / Validation / Test**.
- Chọn tham số giao dịch trên **Validation**, sau đó khóa tham số và đánh giá một lần trên **Test**.
- Mô hình chính: **SARIMAX–LSTM**.
- Mô hình đối chứng: Naive, Ridge, RandomForest, LSTM và SARIMAX đơn lẻ.
- RandomForest chỉ dùng làm benchmark và feature importance, không phải thành phần của mô hình lai chính.
- SARIMAX sử dụng `seasonal_order = [1, 0, 1, 5]` để kiểm tra hiệu ứng chu kỳ tuần giao dịch.
- Backtest có tính `transaction_cost` và `slippage`, mặc định tổng chi phí hiệu dụng là 0.10% cho mỗi lần thay đổi vị thế.
- Có Diebold-Mariano test, extended trading metrics, monthly/annual/regime analysis và walk-forward robustness check.

## 2. Cấu trúc thư mục

```text
GoldForecast-Hybrid-System/
├── configs/
│   └── config.yaml
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   ├── 03_train_evaluate.py
│   └── run_all.py
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── trading/
│   ├── utils/
│   └── pipeline.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── artifacts/
│   ├── metrics/
│   ├── plots/
│   └── models/
├── main.py
├── requirements.txt
├── README.md
├── PROJECT_NOTES.md
└── SUBMISSION_GUIDE.md
```

## 3. Cài đặt môi trường

```powershell
cd "D:\Đồ án 1\GoldForecast-Hybrid-System"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu máy đang dùng Python 3.13 và TensorFlow chưa hỗ trợ, hệ thống sẽ tự chuyển sang fallback neural regressor cho phần LSTM. Để chạy đúng Keras/TensorFlow LSTM, nên dùng môi trường Python tương thích TensorFlow, ví dụ Python 3.10 hoặc 3.11.

## 4. Cách chạy

Chạy toàn bộ pipeline:

```powershell
python .\scripts\run_all.py
```

Hoặc chạy từng bước:

```powershell
python .\scripts\01_download_data.py
python .\scripts\02_prepare_data.py
python .\scripts\03_train_evaluate.py
```

## 5. Pipeline thực nghiệm

```text
Download data
→ Clean & align data
→ Feature engineering
→ Data filtering & feature selection
→ Train / Validation / Test split
→ Fit models on Train and predict Validation
→ Select alpha and trading parameters on Validation
→ Refit models on Train + Validation
→ Predict final Test period
→ Forecast evaluation
→ Trading backtest with frozen validation-selected parameters
→ Statistical tests, extended metrics, walk-forward check and plots
```

## 6. Công thức mô hình lai

```text
SARIMAX_return = SARIMAX(target_log_return, exog=X)
residual       = actual_return - SARIMAX_return
LSTM_residual  = LSTM(residual_history)
final_return   = SARIMAX_return + alpha * LSTM_residual
forecast_close = close_lag_1 * exp(final_return)
```

Trong thực nghiệm chính, `alpha` mặc định là 1.0 nếu không bật tối ưu alpha. Nếu bật `hybrid.optimize_alpha`, alpha chỉ được chọn trên tập Validation.

## 7. Các file kết quả quan trọng

```text
data/predictions/validation_predictions.csv
data/predictions/test_predictions.csv
artifacts/metrics/forecast_metrics.csv
artifacts/metrics/return_metrics.csv
artifacts/metrics/validation_sweep_metrics.csv
artifacts/metrics/selected_trading_params.csv
artifacts/metrics/final_test_trading_metrics.csv
artifacts/metrics/dm_test_results.csv
artifacts/metrics/extended_trading_metrics.csv
artifacts/metrics/monthly_returns.csv
artifacts/metrics/annual_stats.csv
artifacts/metrics/regime_metrics.csv
artifacts/metrics/sarimax_diagnostics.csv
artifacts/metrics/methodology_notes.txt
artifacts/metrics/walk_forward/wf_forecast_summary.csv
artifacts/metrics/walk_forward/wf_trading_summary.csv
artifacts/plots/forecast_comparison.png
artifacts/plots/equity_curves.png
artifacts/plots/pipeline_diagram.png
artifacts/models/model_manifest.json
```

Kết quả chính nên lấy từ `forecast_metrics.csv`, `return_metrics.csv`, `final_test_trading_metrics.csv` và `dm_test_results.csv`. File `validation_sweep_metrics.csv` dùng để chứng minh quy trình chọn tham số trên validation, không dùng làm kết quả cuối cùng.

## 8. Ghi chú phương pháp luận

- CPI và Fed Rate là biến vĩ mô tần suất thấp. Việc căn chỉnh sang dữ liệu ngày bằng forward-fill chỉ là xấp xỉ; hướng cải thiện tốt hơn là căn chỉnh theo ngày công bố thực tế.
- Walk-forward validation được dùng như robustness check, không dùng để chọn tham số giao dịch chính.
- Backtest là mô phỏng nghiên cứu, chưa phải hệ thống giao dịch thực tế hoặc bot giao dịch tự động.
- Kết quả giao dịch phụ thuộc vào regime thị trường, chi phí giao dịch, trượt giá và quy tắc vào/ra lệnh.
- Không nên diễn giải SARIMAX–LSTM là mô hình luôn vượt trội tuyệt đối; kết luận phù hợp hơn là mô hình có kết quả cạnh tranh và có tiềm năng trong phạm vi dữ liệu nghiên cứu.

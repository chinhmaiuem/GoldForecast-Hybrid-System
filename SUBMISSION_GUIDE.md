# Submission Guide

## Tên hệ thống

**GoldForecast-Hybrid System**

## Tên đề tài

**Nghiên cứu mô hình lai SARIMAX–LSTM trong dự báo giá vàng quốc tế và mô phỏng chiến lược giao dịch ngắn hạn**

## Mô hình chính

Mô hình chính của hệ thống là **SARIMAX–LSTM**. SARIMAX dự báo `target_log_return` với biến ngoại sinh, sau đó LSTM học residual của SARIMAX. Dự báo cuối cùng được quy đổi về giá đóng cửa để đánh giá sai số dự báo giá.

## Kết quả nên báo cáo

Sử dụng các file sau làm kết quả chính:

```text
artifacts/metrics/forecast_metrics.csv
artifacts/metrics/return_metrics.csv
artifacts/metrics/final_test_trading_metrics.csv
artifacts/metrics/dm_test_results.csv
artifacts/metrics/sarimax_diagnostics.csv
artifacts/metrics/walk_forward/wf_forecast_summary.csv
artifacts/metrics/walk_forward/wf_trading_summary.csv
```

`validation_sweep_metrics.csv` và `selected_trading_params.csv` dùng để chứng minh tham số được chọn trên validation, không dùng như kết quả test chính.

## Câu kết luận khuyến nghị

SARIMAX–LSTM cho kết quả dự báo cạnh tranh trên tập test chính và có khả năng tạo tín hiệu giao dịch có giá trị trong mô phỏng ngắn hạn. Tuy nhiên, chiến lược Buy & Hold vẫn là benchmark mạnh trong giai đoạn giá vàng tăng mạnh, còn walk-forward validation cho thấy hiệu quả của mô hình phụ thuộc vào regime thị trường. Vì vậy, SARIMAX–LSTM nên được xem là hướng tiếp cận có tiềm năng, chưa phải mô hình vượt trội tuyệt đối trong mọi điều kiện thị trường.

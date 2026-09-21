# Project Notes

## 1. Mục tiêu hệ thống

GoldForecast-Hybrid System được xây dựng nhằm nghiên cứu mô hình lai SARIMAX–LSTM cho bài toán dự báo giá vàng quốc tế và mô phỏng chiến lược giao dịch ngắn hạn. Hệ thống tập trung vào tính đúng đắn của quy trình thực nghiệm: dữ liệu được chia theo thời gian, tham số giao dịch được chọn trên validation và kết quả cuối cùng được đánh giá trên test.

## 2. Vai trò của các mô hình

- **SARIMAX–LSTM**: mô hình nghiên cứu chính.
- **SARIMAX**: mô hình thống kê khai thác thành phần tuyến tính và biến ngoại sinh.
- **LSTM**: mô hình học phần residual phi tuyến còn lại sau SARIMAX.
- **Naive, Ridge, RandomForest, LSTM và SARIMAX đơn lẻ**: các mô hình đối chứng.
- **RandomForest**: dùng cho benchmark và feature importance, không nằm trong công thức của mô hình lai SARIMAX–LSTM.

## 3. Quy trình hạn chế backtest overfitting

Hệ thống không chọn ngưỡng giao dịch trực tiếp trên test. Quy trình được thiết kế như sau:

```text
Train       → huấn luyện mô hình
Validation  → chọn threshold, strategy parameters và alpha nếu bật tối ưu
Test        → đánh giá cuối cùng sau khi đã khóa tham số
```

Cách làm này giúp kết quả test đáng tin cậy hơn so với việc sweep tham số trực tiếp trên test rồi chọn cấu hình có lợi nhuận cao nhất.

## 4. Các hạn chế đã được ghi nhận

- CPI và Fed Rate là biến vĩ mô tần suất thấp; forward-fill sang daily chỉ là xấp xỉ.
- SARIMAX order mặc định là cấu hình thực nghiệm; có thể bật AIC grid search nhưng thời gian chạy sẽ tăng.
- Walk-forward robustness check được dùng để kiểm tra độ ổn định qua nhiều giai đoạn, không dùng để tối ưu tham số cuối cùng.
- Sharpe ratio mặc định sử dụng `risk_free_daily = 0.0`.
- Backtest đã cộng thêm slippage, nhưng vẫn là mô phỏng nghiên cứu, không phản ánh đầy đủ toàn bộ điều kiện giao dịch thực tế.
- Seasonal order `[1,0,1,5]` được dùng để kiểm tra hiệu ứng chu kỳ tuần giao dịch; seasonal yearly `[1,0,1,252]` chỉ nên xem là thử nghiệm mở rộng.

## 5. Cách diễn giải kết quả an toàn

Không nên viết:

> SARIMAX–LSTM luôn là mô hình tốt nhất hoặc chiến lược giao dịch tối ưu.

Nên viết:

> Trong phạm vi dữ liệu và quy trình đánh giá của đề tài, SARIMAX–LSTM cho kết quả dự báo cạnh tranh với các mô hình đối chứng. Khi chuyển sang mô phỏng giao dịch, hiệu quả phụ thuộc mạnh vào quy tắc sinh tín hiệu, tham số chiến lược, chi phí giao dịch và regime thị trường. Kết quả walk-forward được sử dụng để kiểm tra thêm độ ổn định của hệ thống qua các giai đoạn khác nhau.

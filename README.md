# Ứng dụng Streamlit dự báo rủi ro khách hàng

Ứng dụng này được chuyển từ notebook `qtrr.ipynb`. Notebook sử dụng mô hình **Logistic Regression** để phân loại biến mục tiêu `PD` dựa trên 24 biến đầu vào:

`TC1, TC2, TC3, TC4, TC5, NL1, NL2, NL3, NL4, DK1, DK2, DK3, DK4, DK5, V1, V2, V3, V4, V5, V6, TS1, TS2, TS3, TS4`.

Notebook không dùng scaler, encoder hoặc bước tạo biến phái sinh, nên app giữ nguyên các biến số đầu vào theo dữ liệu mẫu.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc dữ liệu đầu vào

File dữ liệu nên ở định dạng `.csv`, `.xlsx` hoặc `.xls`.

Các cột bắt buộc để huấn luyện:

| Cột | Vai trò |
|---|---|
| `TC1` đến `TC5` | Biến đầu vào |
| `NL1` đến `NL4` | Biến đầu vào |
| `DK1` đến `DK5` | Biến đầu vào |
| `V1` đến `V6` | Biến đầu vào |
| `TS1` đến `TS4` | Biến đầu vào |
| `PD` | Biến mục tiêu |

File mẫu `5c.csv` có 150 dòng và 27 cột. Ngoài các cột dùng trong mô hình, file còn có `Dấu thời gian` và `NN`; hai cột này không được notebook đưa vào mô hình.

## Các tab trong ứng dụng

1. **Tổng quan dữ liệu**: xem số dòng, số cột, dung lượng file, dữ liệu thô và thống kê mô tả các biến mô hình.
2. **Trực quan hóa dữ liệu**: vẽ 4 biểu đồ cân đối, ưu tiên biến mục tiêu `PD`.
3. **Kết quả huấn luyện & kiểm định mô hình**: hiển thị Accuracy, Precision, Recall, F1, ROC-AUC, ma trận nhầm lẫn, classification report và bảng dự báo tập kiểm định.
4. **Sử dụng mô hình**: dự báo bằng cách nhập trực tiếp từng biến hoặc tải file dự báo hàng loạt.

## Ghi chú kỹ thuật

- Notebook gốc dùng `train_test_split(test_size=0.2, random_state=23)` và `LogisticRegression()` mặc định.
- App cho phép chỉnh `test_size`, `random_state`, `C`, `max_iter`, `solver`; giá trị mặc định bám theo notebook/scikit-learn.
- Mô hình chỉ được huấn luyện khi bấm nút ở sidebar. Kết quả được lưu trong `st.session_state` để không train lại khi chuyển tab.
- Khuyến nghị dùng Streamlit bản mới (`>=1.55`) để hỗ trợ tốt layout, container động và trải nghiệm giao diện hiện đại.

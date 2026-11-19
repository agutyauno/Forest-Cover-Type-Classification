# Mô hình

Các file `.joblib` và `.npz` không đưa vào git vì kích thước lớn; chạy lại notebook sẽ tạo lại chúng
(`random_state` cố định).

| File | Trong git? | Tạo ở | Nội dung |
| :--- | :---: | :--- | :--- |
| `rf_covertype_final.joblib` | Không (≈ 470 MB) | Bước 4.7 | RF học trên mẫu 15% (69.720 mẫu), 300 cây, `max_depth=20` |
| `rf_covertype_final_v2.joblib` | Không (≈ 76 MB) | Bước 5B.5 | Mô hình chủ cho Bước 6: 348.608 mẫu, 150 cây, 26 đặc trưng (F2) |
| `splits_v2.npz` | Không (≈ 4,5 MB) | Bước 5B.5 | Chỉ số dòng của `idx_train`, `idx_val` (D_val), `idx_final` (D_final) |
| `rf_v2_config.json` | Có | Bước 5B.5 | Siêu tham số, danh sách đặc trưng, vai trò của các tập |

Phạm vi áp dụng của mô hình v2: xem mục 5B.8 của notebook. Kết quả E6 cho thấy hiệu năng giảm mạnh khi kiểm thử trên vùng
hoang dã không có trong dữ liệu huấn luyện.

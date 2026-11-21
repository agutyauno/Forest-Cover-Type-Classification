# Mô hình

Các file `.joblib` và `.npz` không đưa vào git vì kích thước lớn. Chạy lại notebook sẽ tạo lại chúng (`random_state` cố định).

| File | Trong git? | Tạo ở | Nội dung |
| :--- | :---: | :--- | :--- |
| `rf_covertype_final.joblib` | Không (≈ 107 MB) | Bước 9 | Mô hình cuối: RF 200 cây, 57 đặc trưng (không có `Wilderness_Area`), học trên 348.608 mẫu (D_pool) |
| `splits.npz` | Không (≈ 4,5 MB) | Bước 9 | Chỉ số dòng của `idx_train` (D_pool), `idx_val` (D_val), `idx_final` (D_final) |
| `rf_final_config.json` | Có | Bước 9 | Siêu tham số, `feature_spec` cho `build_features()`, danh sách đặc trưng, vai trò của các tập |

Phạm vi áp dụng: xem Bước 11 của notebook. Mô hình chỉ được đánh giá bằng phép chia ngẫu nhiên bên trong 4 vùng nghiên cứu; khả năng chuyển sang vùng khác không được đo.

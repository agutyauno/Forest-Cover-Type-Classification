# KẾ HOẠCH THÍ NGHIỆM VÀ QUY TRÌNH ĐÁNH GIÁ
## Đề tài: Phân loại kiểu thảm rừng từ dữ liệu địa hình (Forest Cover Type Classification)

> **Trạng thái**: tài liệu thiết kế (mới thêm sau rà soát). Các thí nghiệm dưới đây biến những **[GIẢ THUYẾT]** trong bộ tài liệu thành các phép đo có thể kiểm chứng.
>
> **Cập nhật tiến độ**: E0–E3 và E5 sơ bộ đã chạy trên mẫu 15% (notebook Bước 4). E4, E2b (tổ hợp khoảng cách), E3-D (Code + Zones) và E5 mở rộng đã chạy ở **Bước 5B** với phân chia mới: D_final 20% (rút từ hold-out cũ, đánh giá một lần) / D_val 20% / D_pool 60%. E6 (leave-one-wilderness-out) đã chạy ở mục 5B.7. Kết quả và hạn chế xem mục 5B.8 của notebook. E7 chưa chạy.
>
> **Tài liệu liên quan**: [Kế hoạch triển khai](KE_HOACH_TRIEN_KHAI.md) · [Chiến lược lấy mẫu](../03_co_so_ly_thuyet/CHIEN_LUOC_LAY_MAU_PHAN_TANG.md) · [Kỹ thuật đặc trưng](../03_co_so_ly_thuyet/KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md) · [Trích xuất luật](../03_co_so_ly_thuyet/PHUONG_PHAP_TRICH_XUAT_LUAT.md)

---

## 1. MỤC ĐÍCH

Trả lời bằng số đo các câu hỏi mà tài liệu thiết kế mới chỉ nêu giả thuyết: trọng số lớp có thật sự giúp lớp 4 không? Biểu diễn thổ nhưỡng nào tốt hơn? Các đặc trưng sinh thái có đóng góp không? Mẫu 15% làm mất bao nhiêu hiệu năng so với dùng nhiều dữ liệu hơn? Luật trích xuất mô tả được bao nhiêu hành vi của Random Forest?

---

## 2. QUY TRÌNH DỮ LIỆU VÀ ĐÁNH GIÁ

### 2.1. Phân chia dữ liệu
| Tập | Cách tạo | Quy mô xấp xỉ | Dùng để |
| :--- | :--- | :---: | :--- |
| **Mẫu 15%** | Lấy mẫu phân tầng theo nhãn từ 581.012 dòng | 87.152 | Toàn bộ pha thử nghiệm |
| **Train** | 80% mẫu 15%, `stratify=y` | ≈ 69.721 | Huấn luyện; tinh chỉnh bằng CV **chỉ trên Train** |
| **Test** | 20% mẫu 15%, `stratify=y` | ≈ 17.431 (lớp 4: ≈ 82) | Đánh giá chính |
| **Hold-out lớn** | 85% còn lại của dữ liệu gốc | ≈ 493.860 (lớp 4: ≈ 2.335) | Đánh giá cuối (một lần) và bể truy vấn cho cây đại diện |

- Hold-out lớn chia thành hai phần **rời nhau**: *bể truy vấn* (tối đa vài chục nghìn mẫu, dùng để huấn luyện cây đại diện trên dự đoán của RF) và *phần đánh giá cuối*.
- Không dùng hold-out để chọn siêu tham số hay cấu hình. Chỉ nhìn kết quả hold-out sau khi đã chốt cấu hình.
- Mọi tập đi qua **cùng một hàm biến đổi đặc trưng**; trọng số lớp chỉ tính trên Train.

### 2.1b. Phân chia dữ liệu của Bước 5B (dùng cho E4–E7) — **[SỐ ĐO]**
Sau E0–E3, dữ liệu được chia lại để có tập so sánh lớn hơn và một tập đánh giá cuối chưa từng dùng để chọn cấu hình:

| Tập | Cách tạo | Quy mô (lớp 4) | Dùng để |
| :--- | :--- | :---: | :--- |
| **D_final** | 20% toàn bộ, phân tầng, **chỉ rút từ hold-out cũ** | 116.202 (549) | Đánh giá một lần sau khi chốt cấu hình; đo Fidelity ở E7 |
| **D_val** | 20% toàn bộ, từ phần còn lại | 116.202 (550) | So sánh cấu hình ở E4, E2b, E5; bể truy vấn cho cây đại diện ở E7 |
| **D_pool** | 60% còn lại | 348.608 (1.648) | Huấn luyện |

Chỉ số dòng của các tập được lưu trong `models/splits_v2.npz` (ngoài git; tạo lại khi chạy notebook).

### 2.2. Lặp và độ bất định
- Lặp tối thiểu 3 seed (khác nhau ở lấy mẫu, chia tập, RF) hoặc `StratifiedKFold` 5 phần trên Train; báo cáo trung bình ± độ lệch chuẩn.
- Chênh lệch nhỏ hơn độ lệch chuẩn giữa các seed được coi là **không khác biệt có ý nghĩa**.

### 2.3. Chỉ số báo cáo
Accuracy; Macro-F1; Precision/Recall/F1 từng lớp (đặc biệt lớp 4); ma trận nhầm lẫn; thời gian huấn luyện; RAM đỉnh; kích thước mô hình.

### 2.4. Khoảng tin cậy cho lớp hiếm
Recall lớp 4 trên Test dựa trên ≈ 82 mẫu nên rất bất định: Recall ≈ 0.80 có khoảng tin cậy 95% khoảng ±9 điểm phần trăm. Trên hold-out lớn (≈ 2.335 mẫu lớp 4) khoảng này thu hẹp còn khoảng ±1.6 điểm. Luôn báo cáo khoảng tin cậy Wilson hoặc bootstrap kèm Recall lớp 4.

---

## 3. DANH MỤC THÍ NGHIỆM

| ID | Câu hỏi | Biến thay đổi | Giữ cố định | Ghi chú |
| :---: | :--- | :--- | :--- | :--- |
| **E0** | Baseline là bao nhiêu? | RF mặc định (`n_estimators=150`), 54 cột gốc, `class_weight=None` | Mẫu 15% | Mốc so sánh cho mọi thí nghiệm sau |
| **E1** | Trọng số lớp có giúp lớp 4 không? | `class_weight` ∈ {`None`, `'balanced'`, `'balanced_subsample'`}; tuỳ chọn `BalancedRandomForest` | Đặc trưng gốc | Theo dõi cả Precision lớp 4 (trọng số có thể làm giảm) |
| **E2** | Đặc trưng sinh thái nào đáng giữ? | Thêm lần lượt: $D_{\text{hydro}}$, $\text{Elevation}_{\text{water}}$, Aspect sin/cos, $\Delta$Hillshade, tổ hợp khoảng cách; thử cả bỏ lần lượt | Cấu hình tốt nhất của E1 | Kèm Permutation Importance |
| **E3** | Biểu diễn đất nào tốt hơn? | (A) One-Hot, (B) `Soil_Type_Code`, (C) Climatic + Geologic Zone, (D) Code + Zones | Cấu hình tốt nhất của E1–E2 | Kiểm chứng giả thuyết "pha loãng" |
| **E4** | Mẫu 15% mất bao nhiêu hiệu năng? | Kích thước train: 5% / 10% / 15% / 30% / 50% / 100% của pool huấn luyện | **Test cố định chung**: 20% toàn bộ dữ liệu, `stratify=y` (≈ 116.200; lớp 4 ≈ 549) | Learning curve; kèm thời gian và RAM từng mức |
| **E5** | Siêu tham số nào hợp lý? | `max_depth` ∈ {10, 15, 20, None}; `n_estimators` ∈ {100, 150, 300}; `max_features`; `min_samples_leaf` | Cấu hình tốt nhất của E1–E3 | Chỉ CV trên Train |
| **E6** | Mô hình có tổng quát hoá sang vùng khác không? | Leave-one-wilderness-out (huấn luyện 3 vùng `Wilderness_Area`, kiểm thử vùng còn lại) | Cấu hình đã chốt | Dữ liệu UCI **không có tọa độ**, đây chỉ là phép kiểm tra thay thế cho tự tương quan không gian `[16]`. Một số lớp có thể vắng ở một số vùng: kiểm tra trước và chỉ tính thước đo trên lớp có mặt |
| **E7** | Cây đại diện xấp xỉ RF tốt đến đâu? | Nhãn huấn luyện (RF vs nhãn thật); `max_depth` 3–6; `min_samples_leaf` 50–500 | RF đã chốt | Đo Fidelity, accuracy, số lớp có luật, độ dài luật |

---

## 4. MẪU BẢNG BÁO CÁO

| Cấu hình | Accuracy (TB ± sd) | Macro-F1 | Recall lớp 4 [KTC 95%] | Precision lớp 4 | Thời gian | RAM đỉnh |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| E0 baseline | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] |
| E1 `balanced` | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] | [CẦN ĐO] |
| … | | | | | | |

---

## 5. QUY TẮC RA QUYẾT ĐỊNH

1. Chọn cấu hình theo Macro-F1 và Recall lớp 4 trên CV/Test, **không** dựa trên hold-out lớn.
2. Khi hai cấu hình không khác biệt có ý nghĩa, chọn cấu hình đơn giản hơn (ít đặc trưng hơn, mô hình nhỏ hơn).
3. Sau khi chốt cấu hình, đánh giá **một lần** trên phần đánh giá cuối của hold-out lớn.
4. Thay toàn bộ **[KỲ VỌNG]**/**[GIẢ THUYẾT]**/**[CẦN ĐO]** trong `docs/` bằng số đo thật; nếu kết quả trái giả thuyết, ghi rõ và sửa lập luận tương ứng.

---

## 6. DANH SÁCH KIỂM TRA TRƯỚC KHI CHẠY

- [x] Đặt `random_state`, ghi phiên bản thư viện (`requirements.txt`). Cấu hình Colab: **[CẦN KIỂM TRA]** — notebook mới chạy trên máy cục bộ.
- [x] Đối chiếu số lượng từng lớp với bảng 7 lớp; xác nhận $581.012 \times 55$ và 0 giá trị thiếu.
- [x] Kiểm tra quy ước dấu của `Vertical_Distance_To_Hydrology` trước khi dùng $\text{Elevation}_{\text{water}}$.
- [x] Một hàm biến đổi đặc trưng dùng chung cho Train/Test/hold-out/hàm tương tác.
- [x] Trọng số lớp chỉ tính trên Train; hold-out không dùng để tinh chỉnh.
- [x] Lưu kết quả ra CSV trong `results/tables/`. Các bảng E4–E6 có cột seed; `ablation_results.csv` (E0–E3) chỉ chạy một seed (42).

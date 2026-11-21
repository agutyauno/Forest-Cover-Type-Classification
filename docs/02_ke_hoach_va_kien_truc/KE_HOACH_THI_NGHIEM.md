# KẾ HOẠCH THÍ NGHIỆM VÀ QUY TRÌNH ĐÁNH GIÁ
## Đề tài: Phân loại kiểu thảm rừng từ dữ liệu địa hình (Forest Cover Type Classification)

> **Trạng thái**: tài liệu thiết kế (mới thêm sau rà soát). Các thí nghiệm dưới đây biến những **[GIẢ THUYẾT]** trong bộ tài liệu thành các phép đo có thể kiểm chứng.
>
> **Cập nhật tiến độ**: E0–E5 đã chạy trong notebook (một quy trình 11 bước, phân chia dữ liệu ở mục 2.1); E6 đã được loại khỏi phạm vi (xem bảng dưới). Kết quả, quyết định và hạn chế: Bước 11 của notebook; bảng số đo trong `results/tables/`. E7 chưa chạy.
>
> **Tài liệu liên quan**: [Kế hoạch triển khai](KE_HOACH_TRIEN_KHAI.md) · [Chiến lược lấy mẫu](../03_co_so_ly_thuyet/CHIEN_LUOC_LAY_MAU_PHAN_TANG.md) · [Kỹ thuật đặc trưng](../03_co_so_ly_thuyet/KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md) · [Trích xuất luật](../03_co_so_ly_thuyet/PHUONG_PHAP_TRICH_XUAT_LUAT.md)

---

## 1. MỤC ĐÍCH

Trả lời bằng số đo các câu hỏi mà tài liệu thiết kế mới chỉ nêu giả thuyết: trọng số lớp có thật sự giúp lớp 4 không? Biểu diễn thổ nhưỡng nào tốt hơn? Các đặc trưng sinh thái có đóng góp không? Mẫu 15% làm mất bao nhiêu hiệu năng so với dùng nhiều dữ liệu hơn? Luật trích xuất mô tả được bao nhiêu hành vi của Random Forest?

---

## 2. QUY TRÌNH DỮ LIỆU VÀ ĐÁNH GIÁ

### 2.1. Phân chia dữ liệu (đã triển khai)
| Tập | Cách tạo | Quy mô (lớp 4) | Dùng để |
| :--- | :--- | :---: | :--- |
| **D_final** | 20% toàn bộ, `stratify=y` | 116.202 (549) | Đánh giá **một lần** sau khi chốt cấu hình; đo Fidelity ở E7 |
| **D_val** | 20% toàn bộ, `stratify=y` | 116.202 (550) | So sánh cấu hình ở E0–E5; bể truy vấn cho cây đại diện ở E7 |
| **D_pool** | 60% còn lại | 348.608 (1.648) | Bể huấn luyện (E4, E5, mô hình cuối) |
| **S15** | 15% toàn bộ, rút phân tầng từ D_pool | 87.152 | Huấn luyện ở pha khảo sát E0–E3 |

- Không dùng D_final để chọn siêu tham số hay cấu hình. Chỉ nhìn kết quả D_final sau khi đã chốt cấu hình.
- Mọi tập đi qua **cùng một hàm biến đổi đặc trưng** (`build_features()`); trọng số lớp do scikit-learn tính trên dữ liệu huấn luyện.
- Chỉ số dòng của các tập được lưu trong `models/splits.npz` (ngoài git; tạo lại khi chạy notebook).
- *Thiết kế ban đầu* (mẫu 15% chia Train 80% / Test 20%, hold-out 85%) đã được thay bằng phân chia trên. Lý do: Test cũ chỉ có ≈ 82 mẫu lớp 4, và chính Test đó vừa dùng để chọn cấu hình vừa dùng để báo cáo kết quả.

### 2.2. Lặp và độ bất định
- Lặp tối thiểu 3 seed (khác nhau ở lấy mẫu, chia tập, RF) hoặc `StratifiedKFold` 5 phần trên Train; báo cáo trung bình ± độ lệch chuẩn.
- Chênh lệch nhỏ hơn độ lệch chuẩn giữa các seed được coi là **không khác biệt có ý nghĩa**.

### 2.3. Chỉ số báo cáo
Accuracy; Macro-F1; Precision/Recall/F1 từng lớp (đặc biệt lớp 4); ma trận nhầm lẫn; thời gian huấn luyện; RAM đỉnh; kích thước mô hình.

### 2.4. Khoảng tin cậy cho lớp hiếm
Với ≈ 82 mẫu lớp 4 (Test của thiết kế ban đầu), Recall ≈ 0.80 có khoảng tin cậy 95% khoảng ±9 điểm phần trăm. Với ≈ 549 mẫu (D_val, D_final), khoảng này thu hẹp còn khoảng ±3 điểm (số đo trên D_final: 0,883 [0,854–0,908]). Luôn báo cáo khoảng tin cậy Wilson hoặc bootstrap kèm Recall lớp 4.

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
| **E6** | Mô hình có tổng quát hoá sang vùng khác không? | Leave-one-wilderness-out (huấn luyện 3 vùng `Wilderness_Area`, kiểm thử vùng còn lại) | Cấu hình đã chốt | **Đã loại khỏi phạm vi.** Phạm vi đánh giá của dự án là dự đoán bên trong 4 vùng nghiên cứu; 4 vùng khác nhau mạnh về đai cao độ và cơ cấu lớp, nên LOWO đo khả năng ngoại suy sang hệ sinh thái khác và không tách được tự tương quan không gian khỏi khác biệt giữa vùng. Phần ablation trên D_val (cột `Wilderness_Area` × 6 tổ hợp khoảng cách) được giữ ở notebook mục 8.1 |
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
- [x] Khảo sát phân bố `Vertical_Distance_To_Hydrology` trước khi dùng $\text{Elevation}_{\text{water}}$ (notebook mục 3.3). `covtype.info` không nêu quy ước dấu, nên diễn giải dấu vẫn là **giả định**; EDA chỉ minh hoạ hệ quả của giả định, không kiểm chứng được nó.
- [x] Một hàm biến đổi đặc trưng dùng chung cho Train/Test/hold-out/hàm tương tác.
- [x] Trọng số lớp chỉ tính trên Train; hold-out không dùng để tinh chỉnh.
- [x] Lưu kết quả ra CSV trong `results/tables/`, mọi bảng có cột seed (3 seed cho mọi so sánh).

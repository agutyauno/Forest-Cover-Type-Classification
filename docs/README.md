# TÀI LIỆU DỰ ÁN: PHÂN LOẠI KIỂU THẢM RỪNG TỪ DỮ LIỆU ĐỊA HÌNH
## Forest Cover Type Classification (UCI Dataset ID: 31)

Chào mừng bạn đến với thư mục tài liệu kỹ thuật của đề tài. Toàn bộ tài liệu được phân loại và cấu trúc thành các thư mục chuyên biệt dưới đây để tiện tra cứu và báo cáo:

---

> **Trạng thái tài liệu**: Bộ tài liệu được viết như **thiết kế trước khi thực hiện**. Từ đợt cập nhật 2 (xem [CHANGELOG_RA_SOAT.md](CHANGELOG_RA_SOAT.md)), các mục đã có số đo được gắn nhãn **[SỐ ĐO]** và dẫn về notebook [`notebooks/Phan_loai_tham_rung.ipynb`](../notebooks/Phan_loai_tham_rung.ipynb) (Bước 1–12, E0–E6); các mục chưa đo vẫn giữ nhãn cũ. Quy ước: **[KỲ VỌNG]** = mục tiêu/ước tính chưa đo; **[GIẢ THUYẾT]** = lập luận cần kiểm chứng bằng thí nghiệm; **[CẦN ĐO]** = giá trị sẽ đo ở giai đoạn thực hiện; **[CẦN XÁC MINH]** = dẫn chứng/số liệu cần đối chiếu nguồn gốc; **[SỐ ĐO]** = kết quả thực nghiệm đã chạy (kèm điều kiện đo). Danh sách thay đổi và việc còn lại: xem [CHANGELOG_RA_SOAT.md](CHANGELOG_RA_SOAT.md).
>
> Cấu trúc toàn bộ repo (notebook, dữ liệu, mô hình, kết quả): xem [README gốc](../README.md).

## MỤC LỤC TÀI LIỆU (DOCUMENTATION INDEX)

```
docs/
├── 01_tong_quan_yeu_cau/
│   └── BOI_CANH_YEU_CAU_MUC_TIEU.md        # Bối cảnh, dữ liệu UCI 31, thách thức và mục tiêu KPIs
├── 02_ke_hoach_va_kien_truc/
│   ├── KE_HOACH_TRIEN_KHAI.md              # Kiến trúc giải pháp kỹ thuật và kế hoạch 8 bước
│   └── KE_HOACH_THI_NGHIEM.md              # Thiết kế thí nghiệm E0–E7, quy trình đánh giá, tiến độ
├── 03_co_so_ly_thuyet/
│   ├── THUAT_TOAN_RANDOM_FOREST.md         # Cơ sở lý thuyết chi tiết thuật toán Random Forest
│   ├── PHUONG_PHAP_TRICH_XUAT_LUAT.md      # Phương pháp luận trích xuất luật If-Then sinh thái học
│   ├── CHIEN_LUOC_LAY_MAU_PHAN_TANG.md     # Nghiên cứu lý thuyết Lấy mẫu phân tầng & Học nhạy cảm chi phí
│   └── KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md # Kỹ thuật tạo đặc trưng sinh thái & Mô hình hóa thổ nhưỡng
├── 04_tai_lieu_tham_khao/
│   └── TAI_LIEU_THAM_KHAO.md               # Danh mục tài liệu tham khảo & Bảng đối chiếu dẫn chứng khoa học
├── ma_nguon_mau/                           # Hàm trích xuất luật đã sửa + bài kiểm thử trên dữ liệu giả lập
├── detai.md                                # Đề bài gốc (yêu cầu: thực hiện trên Google Colab)
├── CHANGELOG_RA_SOAT.md                    # Nhật ký chỉnh sửa sau rà soát + danh mục CẦN XÁC MINH (mới)
└── README.md                               # Trang chỉ mục tài liệu (file này)
```

---

## TÓM TẮT NỘI DUNG CÁC TÀI LIỆU

### 1. [01_tong_quan_yeu_cau/BOI_CANH_YEU_CAU_MUC_TIEU.md](01_tong_quan_yeu_cau/BOI_CANH_YEU_CAU_MUC_TIEU.md)
- **Bối cảnh & Ý nghĩa thực tiễn**: Ứng dụng AI/ML trong điều tra thảm thực vật rừng, phòng chống cháy rừng và bảo tồn đa dạng sinh học tại dãy núi Rocky (Colorado, Hoa Kỳ).
- **Đặc tả dữ liệu**: Bộ dữ liệu UCI 31 (581.012 dòng, 54 đặc trưng) phân loại 7 kiểu thảm rừng.
- **Thách thức cốt lõi**: Chi phí tính toán trên Colab và mất cân bằng lớp cực đoan (Lớp 1 & 2 chiếm 85.22%, Lớp 4 chỉ 0.473%), giả thuyết pha loãng đặc trưng của 40 cột đất One-Hot (cần kiểm chứng bằng ablation).
- **Yêu cầu & Mục tiêu**: Thực thi 100% trên Google Colab, lấy mẫu phân tầng 15% ($n = 87.152$ dòng) cho pha thử nghiệm, trích xuất luật sinh thái If-Then; ngưỡng kỳ vọng tối thiểu Accuracy ≥ 88%, Macro F1 ≥ 0.78, Recall lớp 4 ≥ 75% (**[KỲ VỌNG]**, do nhóm đặt). **[SỐ ĐO]** mô hình cuối trên D_final (chia ngẫu nhiên): Accuracy 0,974, Macro-F1 0,951, Recall lớp 4 0,874; trên vùng chưa thấy (E6) Macro-F1 chỉ ≈ 0,28 — xem mục 3.2 của tài liệu.

### 2. [02_ke_hoach_va_kien_truc/KE_HOACH_TRIEN_KHAI.md](02_ke_hoach_va_kien_truc/KE_HOACH_TRIEN_KHAI.md)
- **Kiến trúc giải pháp**: Sơ đồ luồng dữ liệu (Data Pipeline Flowchart) từ lấy mẫu, tạo đặc trưng đến huấn luyện và trích xuất luật.
- **Kỹ thuật đặc trưng sinh thái**: Khoảng cách Euclid 3D tới nước, cao độ tuyệt đối của nguồn nước mặt gần nhất, góc lượng giác đón nắng Sin/Cos, ΔHillshade, và 3 cách biểu diễn thổ nhưỡng (One-Hot / `Soil_Type_Code` / ELU Zones) được so sánh bằng ablation.
- **Kế hoạch 8 bước chi tiết**: Lộ trình từ nạp dữ liệu, EDA, tiền xử lý, huấn luyện Random Forest nhạy cảm chi phí, đánh giá toàn diện, trích xuất luật sinh thái If-Then đến xây dựng hàm tương tác thực địa.

### 2b. [02_ke_hoach_va_kien_truc/KE_HOACH_THI_NGHIEM.md](02_ke_hoach_va_kien_truc/KE_HOACH_THI_NGHIEM.md) (mới)
- **Thiết kế thí nghiệm**: ablation (trọng số lớp, biểu diễn thổ nhưỡng, đặc trưng sinh thái, kích thước mẫu), learning curve, đánh giá trên hold-out lớn (≈ 493.860 mẫu chưa dùng), khoảng tin cậy cho lớp hiếm, mẫu bảng báo cáo.
- **Tiến độ [SỐ ĐO]**: E0–E6 đã chạy; phân chia dữ liệu D_final / D_val / D_pool / S15 và kết quả tóm tắt được ghi trong tài liệu; E7 (cây đại diện) chưa chạy.

### 3. [03_co_so_ly_thuyet/THUAT_TOAN_RANDOM_FOREST.md](03_co_so_ly_thuyet/THUAT_TOAN_RANDOM_FOREST.md)
- **Lý thuyết nền tảng**: Cây quyết định CART, công thức độ vẩn đục Gini Impurity, hiện tượng phương sai cao.
- **Cơ chế "Hai tầng ngẫu nhiên"**: Bootstrap Aggregating (Bagging) và Feature Randomness ($m = \sqrt{p}$).
- **Kỹ thuật chuyên sâu**: Đánh giá Out-of-Bag (OOB Error), chỉ số tầm quan trọng đặc trưng (Mean Decrease Impurity), cơ chế phạt trọng số cân bằng lớp (`class_weight='balanced'`).
- **Bảng siêu tham số trong Scikit-learn**: Giải thích chi tiết các tham số `n_estimators`, `max_depth`, `class_weight`, `n_jobs`...

### 4. [03_co_so_ly_thuyet/PHUONG_PHAP_TRICH_XUAT_LUAT.md](03_co_so_ly_thuyet/PHUONG_PHAP_TRICH_XUAT_LUAT.md)
- **Bản chất trích xuất luật**: Giải quyết bài toán "Hộp đen" để minh bạch hóa mô hình AI.
- **Quy trình 5 bước**: Huấn luyện cây đại diện (surrogate) trên dự đoán của RF $\to$ Duyệt đồ thị đệ quy DFS $\to$ Đo lường Confidence, Support, Class coverage, Lift, Fidelity (đếm trên mẫu thực, không dùng `tree_.value` có trọng số) $\to$ Lọc Top luật $\to$ Ánh xạ ngữ nghĩa sinh thái.
- **Mã nguồn**: Hàm Python trích xuất luật từ cấu trúc Scikit-learn `tree_`, đã sửa lỗi tính Confidence/Support và kiểm thử trên dữ liệu giả lập.
- **Cấu trúc bảng kết quả**: Bảng 7 quy tắc vàng đại diện cho 7 loài cây có đối chiếu với nghiên cứu thực vật Peet (1981) và Marr (1961).

### 5. [03_co_so_ly_thuyet/CHIEN_LUOC_LAY_MAU_PHAN_TANG.md](03_co_so_ly_thuyet/CHIEN_LUOC_LAY_MAU_PHAN_TANG.md)
- **Cơ sở toán học xác suất**: Không gian mẫu phân hoạch, nguyên lý phân bổ tỷ lệ thuận (Proportional Allocation) Cochran (1977).
- **Định lý phương sai của lấy mẫu phân tầng**: $\text{Var}(\bar{y}_{\text{st}}) \le \text{Var}(\bar{y}_{\text{srs}})$ — định lý cho *ước lượng trung bình*; phạm vi áp dụng cho bài toán phân loại được làm rõ trong tài liệu.
- **Học nhạy cảm chi phí (Cost-Sensitive Learning)**: Tối ưu hàm mục tiêu Gini có trọng số $w_k$ (heuristic `'balanced'` của Scikit-learn), tỷ số trọng số Lớp 4 : Lớp 2 xấp xỉ $103:1$; hiệu quả là **[GIẢ THUYẾT]**.
- **So sánh phương pháp luận**: Đánh giá đối sánh với Simple Random Sampling, Random Undersampling và SMOTE.

### 6. [03_co_so_ly_thuyet/KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md](03_co_so_ly_thuyet/KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md)
- **Giới hạn hình học của Cây quyết định**: Phân tích sự không tương thích của mặt cắt trực giao (Axis-aligned Splits) với khoảng cách chéo không gian.
- **Khoảng cách Euclid & Biến đổi lượng giác**: $D_{\text{hydro}} = \sqrt{H^2 + V^2}$, $\text{Aspect}_{\sin}$, $\text{Aspect}_{\cos}$ bảo toàn tính chất topo của dữ liệu tuần hoàn.
- **Phân tích xác suất về sự pha loãng đặc trưng**: Phân phối siêu bội cho xác suất $\approx 74.7\%$ một nút có $\ge 5/7$ ứng viên là biến đất; tác động thực tế đến hiệu năng là **[GIẢ THUYẾT]**, kiểm chứng bằng ablation.
- **Quy trình xử lý thổ nhưỡng 2 giai đoạn**: Biến đổi Vector-to-Scalar đưa về 1 cột số nguyên danh mục kết hợp Ánh xạ ngữ nghĩa theo hệ thống USFS ELU (Đai khí hậu & Đá mẹ địa chất).

### 7. [04_tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md](04_tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md)
- **Bảng đối chiếu số liệu và nguồn trích dẫn**: Trạng thái xác minh cho từng con số (581.012 dòng, 85.22%, 0.47%, ngưỡng accuracy mục tiêu, các đai cao độ sinh thái).
- **Danh mục 17 tài liệu tham khảo**: Toàn bộ đã được đối chiếu và xác minh nguồn gốc (Blackard 1998, 1999; Peet 1981 với DOI chuẩn; Marr 1961; Breiman 1984, 1996, 2001; Caruana & Niculescu-Mizil 2006 được làm rõ phạm vi nhị phân; Kaggle có URL chính thức và phân tích giao thức; Cochran 1977; Scikit-learn; Strobl 2007; Roberts 2017; Chawla 2002).

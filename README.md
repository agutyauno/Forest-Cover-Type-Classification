# Đồ án Trí tuệ nhân tạo: Phân loại kiểu thảm rừng từ dữ liệu địa hình
### Forest Cover Type Classification (UCI Dataset ID: 31)

Đề tài ứng dụng **Random Forest Classifier** kết hợp **lấy mẫu phân tầng**, **kỹ thuật tạo đặc trưng sinh thái** và trích xuất **luật sinh thái rừng (If-Then)** từ dữ liệu địa hình và thổ nhưỡng của Rừng quốc gia Roosevelt (Roosevelt National Forest), Colorado, Hoa Kỳ.

---

## 📁 Cấu trúc thư mục

```
.
├── notebooks/
│   └── Phan_loai_tham_rung.ipynb   # Notebook chính: Bước 1–11, E0–E5 (đã chạy, có output)
├── data/
│   ├── README.md                   # Nguồn dữ liệu, cách lấy dữ liệu
│   └── raw/                        # covtype.info (trong git); covtype.data(.gz) (ngoài git, notebook tự tải)
├── models/
│   ├── README.md                   # Mô tả từng mô hình
│   ├── rf_final_config.json        # Cấu hình mô hình cuối (trong git)
│   └── *.joblib, splits.npz        # Ngoài git — tạo lại bằng cách chạy notebook
├── results/
│   ├── figures/                    # Biểu đồ do notebook xuất ra (EDA, ma trận nhầm lẫn, E4…)
│   └── tables/                     # Bảng số đo: e0_e3_explore, e4_learning_curve, e5_tuning, e5_location_ablation_val, e5_feature_robustness_val, feature_importance(_groups)
├── docs/                           # Đặc tả, kế hoạch, cơ sở lý thuyết, tài liệu tham khảo (xem docs/README.md)
│   └── ma_nguon_mau/               # Hàm trích xuất luật + kiểm thử trên dữ liệu giả lập
├── AGENTS.md                       # Quy tắc văn phong khoa học của dự án
└── requirements.txt                # Phiên bản thư viện đã dùng để tạo kết quả
```

## ▶️ Cách chạy

**Máy cục bộ**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Phan_loai_tham_rung.ipynb   # rồi chọn "Run All"
```
- Notebook tự tìm thư mục gốc dự án, nên chạy được từ gốc hoặc từ `notebooks/`.
- Nếu `data/raw/covtype.data` không có, notebook tự tải qua `sklearn.datasets.fetch_covtype()` (đã đối chiếu: trùng khớp từng giá trị với file UCI).
- Thời gian chạy toàn bộ đo được ≈ 30 phút (1.776 s) trên máy 24 lõi CPU, 16 GB RAM (E5 ≈ 11 phút, mục 8.1 ≈ 4,5 phút, mục 8.2 ≈ 5 phút, Bước 9 ≈ 2 phút). RAM tiến trình tại các điểm kiểm tra ≈ 1,5 GB khi huấn luyện mô hình cuối (đo rời rạc, không phải đỉnh liên tục).

**Google Colab** (yêu cầu của đề bài, `docs/detai.md`): clone repo, `%cd` vào thư mục repo rồi chạy notebook.
**[CẦN KIỂM TRA]** — notebook mới được chạy trên máy cục bộ. Chưa đo thời gian và RAM trên Colab; Colab miễn phí có ít lõi CPU hơn, nên notebook dự kiến chạy lâu hơn đáng kể.

---

## 📚 Tài liệu (`docs/`)

- **[BOI_CANH_YEU_CAU_MUC_TIEU.md](docs/01_tong_quan_yeu_cau/BOI_CANH_YEU_CAU_MUC_TIEU.md)**: Bối cảnh khoa học, dữ liệu UCI 31, thách thức kỹ thuật, ngưỡng KPI và số đo đối chiếu.
- **[KE_HOACH_TRIEN_KHAI.md](docs/02_ke_hoach_va_kien_truc/KE_HOACH_TRIEN_KHAI.md)**: Kiến trúc giải pháp và kế hoạch 8 bước.
- **[KE_HOACH_THI_NGHIEM.md](docs/02_ke_hoach_va_kien_truc/KE_HOACH_THI_NGHIEM.md)**: Thiết kế thí nghiệm E0–E7, quy trình đánh giá, quy tắc ra quyết định.
- **[THUAT_TOAN_RANDOM_FOREST.md](docs/03_co_so_ly_thuyet/THUAT_TOAN_RANDOM_FOREST.md)**: Cơ sở lý thuyết Random Forest, OOB, độ quan trọng đặc trưng, `class_weight`.
- **[PHUONG_PHAP_TRICH_XUAT_LUAT.md](docs/03_co_so_ly_thuyet/PHUONG_PHAP_TRICH_XUAT_LUAT.md)**: Trích xuất luật If-Then qua cây đại diện, Confidence/Support/Lift/Fidelity.
- **[CHIEN_LUOC_LAY_MAU_PHAN_TANG.md](docs/03_co_so_ly_thuyet/CHIEN_LUOC_LAY_MAU_PHAN_TANG.md)**: Lấy mẫu phân tầng (Cochran 1977) và học nhạy cảm chi phí.
- **[KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md](docs/03_co_so_ly_thuyet/KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md)**: Đặc trưng sinh thái và mô hình hoá thổ nhưỡng theo USFS ELU.
- **[TAI_LIEU_THAM_KHAO.md](docs/04_tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md)**: Bảng đối chiếu số liệu và danh mục 17 tài liệu tham khảo.
- **[CHANGELOG_RA_SOAT.md](docs/CHANGELOG_RA_SOAT.md)**: Nhật ký chỉnh sửa tài liệu và việc còn lại.

---

## 📊 Kết quả chính (số đo thực nghiệm, chi tiết ở notebook Bước 11)

Mô hình cuối: Random Forest 200 cây, 57 đặc trưng (46 cột gốc, sau khi bỏ `Aspect`, 3 cột `Hillshade` và 4 cột `Wilderness_Area`; 5 đặc trưng sinh thái dẫn xuất; 6 tổ hợp khoảng cách), `criterion='entropy'`, `max_features=0.35`, `max_depth=None`, `class_weight=None`, học trên 348.608 mẫu. Siêu tham số và số cây đều do quy tắc chọn tự động (trong ngưỡng 1 SD, chọn phương án đơn giản hơn).

| Đánh giá | Số đo [KTC 95%] |
| :--- | :---: |
| Accuracy trên D_final (chia ngẫu nhiên, n = 116.202) | 0,9747 [0,9739–0,9756] |
| Macro-F1 trên D_final | 0,9523 [0,9491–0,9558] |
| Recall lớp 4 trên D_final (n = 549) | 0,883 [0,854–0,908] |
| Macro-F1 trên D_val khi đưa 4 cột `Wilderness_Area` trở lại (notebook mục 8.1, 3 seed) | 0,9527 ± 0,0008 (so với 0,9526 ± 0,0008) |

**Hạn chế chính**: mọi chỉ số dùng phép chia ngẫu nhiên theo dòng, nên chỉ phản ánh năng lực dự đoán **bên trong 4 vùng nghiên cứu**. Các ô 30 m liền kề có thể nằm ở cả tập huấn luyện lẫn tập đánh giá (tự tương quan không gian), nên các chỉ số có thể lạc quan hơn so với khi áp dụng cho khu vực chưa khảo sát. Dữ liệu không có toạ độ; dự án không đánh giá khả năng chuyển sang vùng mới (thí nghiệm E6 đã được loại khỏi phạm vi, xem `docs/CHANGELOG_RA_SOAT.md`).

---

## 🚀 Trạng thái triển khai
- [x] Tài liệu đặc tả, cơ sở lý thuyết và dẫn chứng khoa học trong `docs/`.
- [x] Notebook một quy trình (Bước 1–11): nạp dữ liệu, EDA, phân chia D_final / D_val / D_pool, mẫu khảo sát S15, E0–E3 (trọng số lớp, đặc trưng sinh thái, thổ nhưỡng), E4 (quy mô dữ liệu), E5 (siêu tham số), ablation cột `Wilderness_Area` và tổ hợp khoảng cách trên D_val (mục 8.1), mô hình cuối, đánh giá một lần trên D_final.
- [x] Bỏ 4 cột `Wilderness_Area` (không khác biệt trên D_val; lý do diễn giải). Giữ 6 tổ hợp khoảng cách (bỏ chúng làm Macro-F1 trên D_val giảm 0,011).
- [ ] Trích xuất luật sinh thái (E7), hàm tương tác thực địa, báo cáo tổng kết.
- [ ] Kiểm tra chạy toàn bộ notebook trên Google Colab.

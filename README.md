# Đồ án Trí tuệ nhân tạo: Phân loại kiểu thảm rừng từ dữ liệu địa hình
### Forest Cover Type Classification (UCI Dataset ID: 31)

Đề tài ứng dụng **Random Forest Classifier** kết hợp **lấy mẫu phân tầng**, **kỹ thuật tạo đặc trưng sinh thái** và trích xuất **luật sinh thái rừng (If-Then)** từ dữ liệu địa hình và thổ nhưỡng của Rừng quốc gia Roosevelt (Roosevelt National Forest), Colorado, Hoa Kỳ.

---

## 📁 Cấu trúc thư mục

```
.
├── notebooks/
│   └── Phan_loai_tham_rung.ipynb   # Notebook chính: Bước 1–12, E0–E6 (đã chạy, có output)
├── data/
│   ├── README.md                   # Nguồn dữ liệu, cách lấy dữ liệu
│   └── raw/                        # covtype.info (trong git); covtype.data(.gz) (ngoài git, notebook tự tải)
├── models/
│   ├── README.md                   # Mô tả từng mô hình
│   ├── rf_final_config.json        # Cấu hình mô hình cuối (trong git)
│   └── *.joblib, splits.npz        # Ngoài git — tạo lại bằng cách chạy notebook
├── results/
│   ├── figures/                    # Biểu đồ do notebook xuất ra (EDA, ma trận nhầm lẫn, E4, E6…)
│   └── tables/                     # Bảng số đo: e0_e3_explore, e4_learning_curve, e5_tuning, e6_lowo, feature_importance
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
- Thời gian chạy toàn bộ đo được ≈ 55 phút trên máy 24 lõi CPU, 16 GB RAM (E6 ≈ 30 phút, E5 ≈ 16 phút). RAM tiến trình tại các điểm kiểm tra ≈ 1,5 GB khi huấn luyện mô hình cuối (đo rời rạc, không phải đỉnh liên tục).

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

## 📊 Kết quả chính (số đo thực nghiệm, chi tiết ở notebook Bước 12)

Mô hình cuối: Random Forest 200 cây, 57 đặc trưng (46 cột gốc, sau khi bỏ `Aspect`, 3 cột `Hillshade` và 4 cột `Wilderness_Area`; 5 đặc trưng sinh thái dẫn xuất; 6 tổ hợp khoảng cách), `criterion='entropy'`, `max_features=0.35`, `max_depth=None`, `class_weight=None`, học trên 348.608 mẫu. Siêu tham số và số cây đều do quy tắc chọn tự động (trong ngưỡng 1 SD, chọn phương án đơn giản hơn).

| Đánh giá | Số đo [KTC 95%] |
| :--- | :---: |
| Accuracy trên D_final (chia ngẫu nhiên, n = 116.202) | 0,9747 [0,9739–0,9756] |
| Macro-F1 trên D_final | 0,9523 [0,9491–0,9558] |
| Recall lớp 4 trên D_final (n = 549) | 0,883 [0,854–0,908] |
| Macro-F1 trên vùng chưa thấy (E6, TB 4 vùng, lớp đã thấy) — mô hình cuối | 0,374 |
| Như trên — mô hình cuối nhưng giữ 4 cột `Wilderness_Area` | 0,277 |
| Như trên — baseline E0 (54 cột gốc, tham số mặc định) | 0,323 |

**Hạn chế chính**: khi kiểm thử trên một vùng hoang dã không có trong dữ liệu huấn luyện (E6), hiệu năng giảm mạnh (0,969 → 0,374). Bỏ 4 cột `Wilderness_Area` gỡ được một phần (từ 0,277 lên 0,374) mà không làm đổi hiệu năng khi chia ngẫu nhiên; phần còn lại phù hợp với việc đai cao độ của cùng một kiểu rừng dịch chuyển giữa các vùng (notebook mục 11.2). Các chỉ số trên D_final chỉ phản ánh năng lực dự đoán **bên trong 4 vùng nghiên cứu**.

---

## 🚀 Trạng thái triển khai
- [x] Tài liệu đặc tả, cơ sở lý thuyết và dẫn chứng khoa học trong `docs/`.
- [x] Notebook một quy trình (Bước 1–12): nạp dữ liệu, EDA, phân chia D_final / D_val / D_pool, mẫu khảo sát S15, E0–E3 (trọng số lớp, đặc trưng sinh thái, thổ nhưỡng), E4 (quy mô dữ liệu), E5 (siêu tham số), mô hình cuối, đánh giá một lần trên D_final, E6 (leave-one-wilderness-out), ablation đặc trưng gắn vị trí (mục 11.1) và chẩn đoán nguyên nhân suy giảm khi chuyển vùng (mục 11.2).
- [x] Ablation đặc trưng gắn vị trí dưới giao thức E6: bỏ `Wilderness_Area` (đã áp dụng). Bỏ 6 tổ hợp khoảng cách: đánh đổi (−0,011 khi chia ngẫu nhiên, +0,018 ở vùng chưa thấy), chưa áp dụng.
- [ ] Trích xuất luật sinh thái (E7), hàm tương tác thực địa, báo cáo tổng kết.
- [ ] Kiểm tra chạy toàn bộ notebook trên Google Colab.

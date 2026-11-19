# Đồ án Trí tuệ nhân tạo: Phân loại kiểu thảm rừng từ dữ liệu địa hình
### Forest Cover Type Classification (UCI Dataset ID: 31)

Đề tài ứng dụng **Random Forest Classifier** kết hợp **lấy mẫu phân tầng**, **kỹ thuật tạo đặc trưng sinh thái** và trích xuất **luật sinh thái rừng (If-Then)** từ dữ liệu địa hình và thổ nhưỡng của Rừng quốc gia Roosevelt (Roosevelt National Forest), Colorado, Hoa Kỳ.

---

## 📁 Cấu trúc thư mục

```
.
├── notebooks/
│   └── Phan_loai_tham_rung.ipynb   # Notebook chính: Bước 1–5B (đã chạy, có output)
├── data/
│   ├── README.md                   # Nguồn dữ liệu, cách lấy dữ liệu
│   └── raw/                        # covtype.info (trong git); covtype.data(.gz) (ngoài git, notebook tự tải)
├── models/
│   ├── README.md                   # Mô tả từng mô hình
│   ├── rf_v2_config.json           # Cấu hình mô hình chủ v2 (trong git)
│   └── *.joblib, splits_v2.npz     # Ngoài git — tạo lại bằng cách chạy notebook
├── results/
│   ├── figures/                    # Biểu đồ do notebook xuất ra (EDA, ma trận nhầm lẫn, E4, E6…)
│   └── tables/                     # Bảng số đo: ablation_results.csv, e4/e5/e6_*.csv
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
- Thời gian chạy toàn bộ đo được ≈ 48 phút trên máy 24 lõi CPU, 16 GB RAM (Bước 5B chiếm phần lớn). RAM tiến trình đo tại các điểm kiểm tra trong notebook cao nhất ≈ 2,2 GB (đo rời rạc, không phải đỉnh liên tục).

**Google Colab** (yêu cầu của đề bài, `docs/detai.md`): clone repo, `%cd` vào thư mục repo rồi chạy notebook.
**[CẦN KIỂM TRA]** — notebook mới được chạy trên máy cục bộ. Chưa đo thời gian và RAM trên Colab; Colab miễn phí có ít lõi CPU hơn, nên Bước 5B dự kiến chạy lâu hơn đáng kể.

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

## 📊 Kết quả chính (số đo thực nghiệm, chi tiết ở notebook mục 5B.8)

| Đánh giá | Mô hình Bước 4.7 (mẫu 15%) | Mô hình v2 (Bước 5B) |
| :--- | :---: | :---: |
| Macro-F1 trên D_final (chia ngẫu nhiên, n = 116.202) [KTC 95%] | 0,834 [0,829–0,840] | 0,954 [0,951–0,957] |
| Recall lớp 4 trên D_final [Wilson 95%] | 0,763 [0,726–0,797] | 0,889 [0,860–0,913] |
| Macro-F1 trên vùng chưa thấy (E6, trung bình 4 vùng, lớp đã thấy) | 0,316 | 0,323 |

**Hạn chế chính**: khi kiểm thử trên vùng hoang dã không có trong dữ liệu huấn luyện (E6), hiệu năng của cả hai mô hình giảm mạnh, và lợi thế của v2 gần như không còn. Các chỉ số trên D_final chỉ phản ánh năng lực dự đoán **bên trong 4 vùng nghiên cứu**.

---

## 🚀 Trạng thái triển khai
- [x] Tài liệu đặc tả, cơ sở lý thuyết và dẫn chứng khoa học trong `docs/`.
- [x] Bước 1–5: nạp dữ liệu, EDA, lấy mẫu phân tầng, đặc trưng sinh thái, ablation E0–E3, E5 sơ bộ trên mẫu 15%.
- [x] Bước 5B: E4 (learning curve), E2b/E3-D (đặc trưng), E5 mở rộng, mô hình chủ v2, đánh giá một lần trên D_final.
- [x] E6: leave-one-wilderness-out.
- [ ] Ablation loại đặc trưng gắn với vị trí (Roadways, Fire_Points) dưới giao thức E6 — **[GIẢ THUYẾT]** có thể cải thiện khả năng chuyển vùng.
- [ ] Bước 6–8: trích xuất luật sinh thái (E7), hàm tương tác thực địa, báo cáo tổng kết.
- [ ] Kiểm tra chạy toàn bộ notebook trên Google Colab.

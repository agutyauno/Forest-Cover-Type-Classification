# NHẬT KÝ CHỈNH SỬA SAU RÀ SOÁT

Bộ tài liệu ban đầu là **thiết kế trước khi thực hiện**. Đợt 1 (mục 1) sửa nội dung thiết kế; đợt 2 (mục 1b) bổ sung số đo thực nghiệm và cấu trúc repo. Nhật ký này ghi lại những gì đã sửa và những gì còn phải làm.

## 1. Đã sửa

| # | Vấn đề | Cách xử lý | File |
| :-: | :--- | :--- | :--- |
| 1 | Hàm trích luật tính Confidence/Support từ `tree_.value` (đã nhân `class_weight`, tổng bằng 1) nên sai; không tính Lift dù tài liệu nêu | Viết lại: dùng `cay.apply()` và đếm trên mẫu thực; thêm Class coverage, Lift, Fidelity; kiểm thử trên dữ liệu giả lập (`ma_nguon_mau/`) | `PHUONG_PHAP_TRICH_XUAT_LUAT.md` |
| 2 | "Surrogate" huấn luyện trên nhãn thật, không đo Fidelity | Cây học theo **dự đoán của RF** trên dữ liệu chưa dùng huấn luyện RF; báo cáo Fidelity; giữ chế độ nhãn thật để so sánh | `PHUONG_PHAP_TRICH_XUAT_LUAT.md`, `KE_HOACH_TRIEN_KHAI.md` |
| 3 | Con số 68,32% (siêu bội) không tái lập | Tính lại: P(X ≥ 5) ≈ **74,7%**; diễn giải đúng mức (chỉ là thành phần tập ứng viên, xác suất một đặc trưng có mặt là m/p) | `KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md` và các file nhắc tới |
| 4 | Lý do lấy mẫu 15%: "RAM > 12GB", định lý phương sai, xác suất Poisson | Bỏ các lập luận sai; ghi rõ định lý chỉ áp dụng cho ước lượng trung bình; n₄ là cố định; dùng learning curve để quyết định kích thước mẫu | `CHIEN_LUOC_LAY_MAU_PHAN_TANG.md`, `BOI_CANH_YEU_CAU_MUC_TIEU.md` |
| 5 | KPI 88–91% / 89–91% mâu thuẫn, nguồn `[9]`, `[10]` đáng ngờ | Đổi thành **ngưỡng tối thiểu kỳ vọng** (≥ 88%, Macro F1 ≥ 0.78, Recall lớp 4 ≥ 75%); đánh dấu `[9]`, `[10]` là **[CẦN XÁC MINH]** | nhiều file, `TAI_LIEU_THAM_KHAO.md` |
| 6 | Gán sai nguồn: King & Zeng cho `'balanced'`; "Breiman chứng minh √p tối ưu" | Chuyển sang Scikit-learn `[14]`; ghi √p là heuristic phổ biến | `CHIEN_LUOC…`, `THUAT_TOAN_RANDOM_FOREST.md` |
| 7 | `Elevation − VDH` gọi là "mực nước ngầm" | Sửa thành cao độ của nguồn nước **mặt** gần nhất; nhắc kiểm tra quy ước dấu | `KY_THUAT…`, `KE_HOACH_TRIEN_KHAI.md`, `README.md` |
| 8 | Mâu thuẫn nội bộ: ngưỡng 3.050m cho cả Spruce/Fir và Krummholz; `Soil_Type29` là đai 7 nhưng ví dụ lại dùng cho Alpine; dải độ cao lệch nhau giữa các file; nhãn đai ELU 6/7 không nhất quán | Dùng một bảng dải độ cao chuẩn (ở `BOI_CANH…`); ví dụ dùng loại đất 38; sửa danh sách `d1`, `d2` theo `covtype.info` (cần đối chiếu) | `KY_THUAT…`, `PHUONG_PHAP…`, `BOI_CANH…` |
| 9 | "Vườn quốc gia Roosevelt" | Sửa thành **Rừng quốc gia** Roosevelt (Roosevelt National Forest) | tất cả file |
| 10 | ΔHillshade dùng nhưng chưa định nghĩa; số cột "15" mâu thuẫn với các đặc trưng thêm | Định nghĩa ΔHillshade = Hillshade_3pm − Hillshade_9am (đề xuất); ghi rõ 15 cột trước khi thêm đặc trưng, ≈ 20–22 sau đó | `KY_THUAT…`, `KE_HOACH_TRIEN_KHAI.md` |
| 11 | Giọng điệu như đã có kết quả ("hội tụ hoàn hảo", "triệt tiêu hoàn toàn", "độ tin cậy > 93%") | Đổi sang giọng giả thuyết, gắn nhãn [KỲ VỌNG]/[GIẢ THUYẾT]/[CẦN ĐO]/[CẦN XÁC MINH]; bảng luật mẫu dùng placeholder | tất cả file |
| 12 | Thiếu: giao thức đánh giá, khoảng tin cậy lớp 4 (≈ 82 mẫu test), ablation, hạn chế (tự tương quan không gian) | Thêm file kế hoạch thí nghiệm; thêm mục Hạn chế/Rủi ro; thêm hold-out lớn (≈ 493.860 mẫu) | `KE_HOACH_THI_NGHIEM.md` (mới), `KE_HOACH_TRIEN_KHAI.md` |
| 13 | Đối chiếu & chuẩn hóa toàn bộ các mục **[CẦN XÁC MINH]** | Xác minh tác giả/tiêu đề `[1]`, `[2]`; sửa DOI `[4]` thành `10.1007/BF00240202`; làm rõ Caruana (2006) `[9]` chỉ đánh giá bài toán nhị phân không phải nguồn cho accuracy 88–91%; bổ sung URL chính thức và phân tích giao thức Kaggle `[10]`; xác nhận khớp dải độ cao Peet (1981) & Marr (1961); chuẩn hóa định nghĩa $d_1$, $d_2$ và cập nhật mô tả 40 loại đất theo đúng `covtype.info` | `TAI_LIEU_THAM_KHAO.md`, `KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md`, `BOI_CANH_YEU_CAU_MUC_TIEU.md`, `PHUONG_PHAP_TRICH_XUAT_LUAT.md`, `KE_HOACH_TRIEN_KHAI.md`, `README.md` |

## 1b. Đợt 2 — Số đo thực nghiệm và cấu trúc repo

| # | Thay đổi | Lý do | File |
| :-: | :--- | :--- | :--- |
| 14 | Sắp xếp repo: `notebooks/`, `data/raw/`, `models/`, `results/{figures,tables}/`; khởi tạo git với `.gitignore` (dữ liệu thô, `.joblib`, `.npz`, `backup/` để ngoài git), `requirements.txt`, `data/README.md`, `models/README.md` | Tách mã, dữ liệu, mô hình và kết quả; không đưa file lớn (≈ 630 MB) vào git; ghi phiên bản thư viện để tái lập | README gốc |
| 15 | Notebook tự tìm thư mục gốc dự án; ghi hình vào `results/figures/`, bảng vào `results/tables/`, mô hình vào `models/`; tự tải dữ liệu qua `sklearn.datasets.fetch_covtype()` khi không có file cục bộ (đã đối chiếu trùng khớp từng giá trị) | Chạy được sau khi clone repo (kể cả Colab) mà không phải sửa đường dẫn tuyệt đối | `notebooks/Phan_loai_tham_rung.ipynb` |
| 16 | Thêm Bước 5B vào notebook: E4, E2b, E3-D, E5 mở rộng, mô hình chủ v2, đánh giá một lần trên D_final, E6 | Thực hiện các thí nghiệm còn thiếu của `KE_HOACH_THI_NGHIEM.md` trước Bước 6 | notebook; `KE_HOACH_THI_NGHIEM.md` |
| 17 | Gắn nhãn **[SỐ ĐO]** và bổ sung số liệu thực nghiệm cạnh các mục **[KỲ VỌNG]/[CẦN ĐO]** (giữ nguyên ngưỡng thiết kế, không ghi đè) | Mục 3.1 của `AGENTS.md`: phân định ngưỡng kỳ vọng với số đo thực nghiệm | `BOI_CANH…` (mục 2.1, 2.2, 3.2, 3.3), `KE_HOACH_TRIEN_KHAI.md`, `KE_HOACH_THI_NGHIEM.md` (mục 2.1b, 6), `THUAT_TOAN_RANDOM_FOREST.md` (bảng siêu tham số), `PHUONG_PHAP…` (mục 6), `docs/README.md` |
| 18 | Sửa mô tả không khớp với mã: nạp dữ liệu bằng `fetch_ucirepo` → thực tế dùng file cục bộ và `fetch_covtype`; tên notebook `Phan_loai_tham_rung_Colab.ipynb` → `notebooks/Phan_loai_tham_rung.ipynb`; đường dẫn `covertype/covtype.info` → `data/raw/covtype.info` | Tài liệu mô tả đúng cách mã đang chạy | `BOI_CANH…`, `KE_HOACH_TRIEN_KHAI.md`, `KY_THUAT…` |
| 19 | `detai.md` đã có trong `docs/`; bỏ ghi chú "không nằm trong gói"; README gốc còn sót "Vườn quốc gia" → "Rừng quốc gia"; số tài liệu tham khảo 13 → 17 | Đính chính | `PHUONG_PHAP…`, `README.md` |
| 20 | Hạn chế mới có số đo: E6 cho Macro-F1 ≈ 0,32 trên vùng chưa thấy (so với 0,86–0,97 khi chia ngẫu nhiên) | Kết quả cần được nêu cạnh mọi chỉ số chia ngẫu nhiên | `KE_HOACH_TRIEN_KHAI.md` (mục 3), `BOI_CANH…` (mục 3.2) |

## 2. Cần người thực hiện xử lý (tiếp theo)

1. ~~Đối chiếu bộ `docs/` với `detai.md`~~ — `detai.md` đã có trong `docs/`. Yêu cầu "thực hiện trên Google Colab" **chưa được kiểm tra**: notebook mới chạy trên máy cục bộ.
2. Các mục **[KỲ VỌNG]/[CẦN ĐO]** liên quan tới Bước 1–5B đã có số đo kèm theo. Còn lại: thời gian/RAM trên Colab; mọi số liệu của Bước 6–8 (Fidelity, luật, hàm tương tác). Hàm trích luật mới chỉ chạy trên dữ liệu giả lập, chưa chạy trên Covertype.
3. **[GIẢ THUYẾT]** cần kiểm chứng trước hoặc cùng Bước 6: loại các đặc trưng gắn với vị trí (Roadways, Fire_Points và tổ hợp của chúng) có thể cải thiện kết quả E6.
4. Các phần lý thuyết (`CHIEN_LUOC…`, `KY_THUAT…`) vẫn trình bày lập luận thiết kế gốc. Khi viết báo cáo cuối, cần đối chiếu các giả thuyết (pha loãng One-Hot, hiệu quả `class_weight`) với kết quả E1–E3 và E5: `class_weight=None` được chọn ở E5 theo quy tắc ít nút nhất, nhưng có Recall lớp 4 thấp hơn `balanced_subsample`.

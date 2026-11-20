# NHẬT KÝ CHỈNH SỬA SAU RÀ SOÁT

Bộ tài liệu ban đầu là **thiết kế trước khi thực hiện**. Đợt 1 (mục 1) sửa nội dung thiết kế; đợt 2 (mục 1b) bổ sung số đo thực nghiệm và cấu trúc repo; đợt 3 (mục 1c) tái cấu trúc notebook; đợt 4 (mục 1d) chạy lại từng bước và bỏ cột `Wilderness_Area`. Số liệu ghi trong mục 1b thuộc bản notebook trước đợt 3; số liệu ở mục 1c thuộc bản trước đợt 4. Nhật ký này ghi lại những gì đã sửa và những gì còn phải làm.

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

## 1c. Đợt 3 — Tái cấu trúc notebook thành một quy trình

| # | Thay đổi | Lý do | File |
| :-: | :--- | :--- | :--- |
| 21 | Notebook viết lại thành 12 bước liền mạch. Phân chia D_final / D_val / D_pool đặt ngay ở Bước 4; mẫu khảo sát S15 (15% toàn bộ) rút từ D_pool cho E0–E3; E0–E6 chạy trên cùng một phép chia, mỗi so sánh lặp 3 seed. Bỏ phân chia Train/Test/Hold-out cũ, mô hình Bước 4.7 và phần CV sơ bộ | Bản trước có hai lớp nối tiếp (pha 15% rồi Bước 5B chia lại dữ liệu), khó theo dõi; Test cũ chỉ có 82 mẫu lớp 4 và vừa dùng để chọn cấu hình vừa dùng để báo cáo | `notebooks/Phan_loai_tham_rung.ipynb` |
| 22 | Gộp hai hàm đặc trưng thành `build_features(df, eco, soil, dist_combo)`; gom cấu hình (seed, số cây, tỷ lệ chia) vào một cell; thêm hàm quy tắc chọn `choose_simplest()` dùng chung cho E1–E5 | Một nguồn duy nhất cho phép biến đổi đặc trưng và quy tắc ra quyết định | notebook |
| 23 | Kết quả mới (số liệu thay đổi vì tập so sánh khác): E3 chọn One-Hot thay vì `Soil_Type_Code`; mô hình cuối 65 đặc trưng, `max_features=0.5`; D_final Macro-F1 0,951 [0,948–0,955]; E6: mô hình cuối 0,277, thấp hơn baseline E0 0,323 | Ghi nhận các kết luận thay đổi so với bản trước | `README.md`, `BOI_CANH…`, `KE_HOACH_*`, `THUAT_TOAN…`, `PHUONG_PHAP…`, `docs/README.md` |
| 24 | Đổi tên đầu ra: `rf_final_config.json`, `splits.npz`, `e0_e3_explore.csv`, `e5_tuning.csv`, `feature_importance.csv`, `eval_confusion_final.png`; gỡ các file đầu ra cũ không còn được tạo (vẫn còn trong lịch sử git) | Tên file khớp với cấu trúc mới | `models/`, `results/` |
| 25 | Sửa lỗi `permutation_importance(n_jobs=-1)` thất bại do phải pickle rừng lớn sang tiến trình con; chuyển sang `n_jobs=1` | Lỗi chỉ xuất hiện ở quy mô dữ liệu đầy đủ | notebook, Bước 10.2 |

## 1d. Đợt 4 — Chạy lại từng bước, bỏ cột Wilderness_Area

| # | Thay đổi | Lý do | File |
| :-: | :--- | :--- | :--- |
| 26 | Báo cáo Precision / Recall / F1 của cả 7 lớp ở mọi thí nghiệm (`eval_preds`, `per_class_table`), D_final kèm KTC bootstrap từng lớp (`results/tables/final_per_class.csv`), E6 kèm F1 từng lớp theo vùng | Các bảng trước chỉ báo cáo lớp 4 và chỉ số tổng hợp | notebook Bước 1, 6–11 |
| 27 | E2 thêm các phương án về hướng sườn / bóng nắng (bỏ `Aspect_Trig`, `Delta_Hillshade`; chỉ giữ dạng dẫn xuất; bỏ hẳn; giữ `Hillshade_Noon`); `build_features()` thêm tham số `drop_raw`. Quy tắc chọn "tổ hợp KC, chỉ dẫn xuất" | Kiểm tra dư thừa giữa cột gốc và cột dẫn xuất (mục 2.5 cũ) | notebook Bước 5–6 |
| 28 | Thêm mục 11.1 (ablation 2×2: `Wilderness_Area` × 6 tổ hợp khoảng cách, chia ngẫu nhiên và LOWO) và 11.2 (cơ cấu lớp theo vùng, ma trận nhầm lẫn LOWO, phân vị cao độ theo vùng × lớp) | Kiểm chứng giả thuyết ở mục 2.3 cũ về đặc trưng gắn vị trí | notebook Bước 11 |
| 29 | **Từ E2 trở đi bỏ 4 cột `Wilderness_Area`** (E0/E1 giữ 54 cột gốc làm baseline); chạy lại E2–E6. Mô hình cuối: 57 đặc trưng, `max_features=0.35`, 200 cây. D_final Macro-F1 0,9523 [0,9491–0,9558] (không khác bản cũ trong KTC); E6 0,374 (bản cũ 0,277; E0 0,323) | Cột WA không đổi hiệu năng khi chia ngẫu nhiên (0,9527 so với 0,9526 ± 0,0008) nhưng giảm LOWO ≈ 0,10, nhất quán ở cả hai chiều thử; luật không còn điều kiện kiểu "nếu thuộc vùng X" | notebook, `models/`, `results/`, `README.md`, `docs/` |
| 30 | Bỏ các tham số quyết định thủ công: mục 8.1 (`PRIORITY`, ưu tiên Precision/Recall lớp hiếm) và `N_TREES_OVERRIDE` ở Bước 9; cấu hình và số cây hoàn toàn theo quy tắc `choose_simplest` | Ở cả hai lần chạy, chênh lệch Precision lớp hiếm giữa các ứng viên chỉ 0,0001, nhỏ hơn nhiều so với dao động giữa các seed | notebook Bước 8–9 |
| 31 | Bước 12 viết lại theo số đo mới. D_final đã được dùng 3 lần (commit `3603bec`, lần chạy từng bước trước khi bỏ WA, lần này); mọi lựa chọn dựa trên D_val hoặc LOWO | Minh bạch việc dùng lại tập đánh giá cuối | notebook Bước 12 |

## 2. Cần người thực hiện xử lý (tiếp theo)

1. ~~Đối chiếu bộ `docs/` với `detai.md`~~ — `detai.md` đã có trong `docs/`. Yêu cầu "thực hiện trên Google Colab" **chưa được kiểm tra**: notebook mới chạy trên máy cục bộ.
2. Các mục **[KỲ VỌNG]/[CẦN ĐO]** liên quan tới Bước 1–5B đã có số đo kèm theo. Còn lại: thời gian/RAM trên Colab; mọi số liệu của Bước 6–8 (Fidelity, luật, hàm tương tác). Hàm trích luật mới chỉ chạy trên dữ liệu giả lập, chưa chạy trên Covertype.
3. ~~Kiểm chứng việc loại đặc trưng gắn vị trí dưới giao thức E6~~ — đã làm ở đợt 4 (mục 1d, #28–29): bỏ `Wilderness_Area` đã áp dụng. Còn để ngỏ: bỏ 6 tổ hợp khoảng cách là một đánh đổi (−0,011 Macro-F1 khi chia ngẫu nhiên, +0,018 ở vùng chưa thấy); `Fire_Road_Sum` vẫn đứng thứ 3 về Permutation importance nên có thể xuất hiện trong luật của cây đại diện.
4. Các phần lý thuyết (`CHIEN_LUOC…`, `KY_THUAT…`) vẫn trình bày lập luận thiết kế gốc. Khi viết báo cáo cuối, cần đối chiếu với số đo: (a) giả thuyết pha loãng One-Hot **không** được ủng hộ (E3: One-Hot cao hơn các phương án gộp); (b) `class_weight='balanced'` tốt nhất trên S15 (E1) nhưng `None` tốt hơn ở quy mô đầy đủ (E5); (c) `class_weight` và `min_samples_leaf` chủ yếu đổi Precision lấy Recall của các lớp hiếm (ví dụ `balanced, min_samples_leaf=4`: Recall lớp 4 0,929 so với 0,884, Macro-F1 −0,024; 1 seed, xem `results/tables/e5_tuning.csv`); (d) `max_depth=20` dự đoán thừa lớp 5 (Recall 0,970, Precision 0,628).
5. ~~Phương án "tổ hợp khoảng cách + bỏ `Aspect_Trig`/`Delta_Hillshade`"~~ — đã thử ở đợt 4 (#27).
6. Số cây (200) do quy tắc chọn dựa trên chênh lệch cỡ nhiễu so với 150 cây (đường cong 1 seed); nếu kích thước file hoặc tốc độ truy vấn ở E7 là vấn đề, có thể cân nhắc lại.

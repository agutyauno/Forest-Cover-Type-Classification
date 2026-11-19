# ĐẶC TẢ BỐI CẢNH, YÊU CẦU VÀ MỤC ĐÍCH ĐỀ TÀI
## Đề tài: Phân loại kiểu thảm rừng từ dữ liệu địa hình (Forest Cover Type Classification)

> **Tài liệu tham chiếu liên quan**:
> - [Kế hoạch triển khai kỹ thuật](../02_ke_hoach_va_kien_truc/KE_HOACH_TRIEN_KHAI.md)
> - [Cơ sở lý thuyết thuật toán Random Forest](../03_co_so_ly_thuyet/THUAT_TOAN_RANDOM_FOREST.md)
> - [Phương pháp luận trích xuất luật sinh thái](../03_co_so_ly_thuyet/PHUONG_PHAP_TRICH_XUAT_LUAT.md)
> - [Nghiên cứu lý thuyết Lấy mẫu phân tầng & Học nhạy cảm chi phí](../03_co_so_ly_thuyet/CHIEN_LUOC_LAY_MAU_PHAN_TANG.md)
> - [Kỹ thuật tạo đặc trưng sinh thái & Mô hình hóa thổ nhưỡng](../03_co_so_ly_thuyet/KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md)
> - [Danh mục tài liệu tham khảo & Dẫn chứng khoa học](../04_tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md)

---

> **Trạng thái tài liệu**: thiết kế trước khi thực hiện. Mọi con số về hiệu năng, thời gian, RAM là **[KỲ VỌNG]** hoặc **[GIẢ THUYẾT]**, sẽ được thay bằng số đo ở Bước 5. Mục **[CẦN XÁC MINH]** là dẫn chứng cần đối chiếu nguồn gốc.

## 1. BỐI CẢNH ĐỀ TÀI VÀ Ý NGHĨA THỰC TIỄN

### 1.1. Bối cảnh khoa học & thực tiễn
Trong lâm nghiệp hiện đại và sinh thái học cảnh quan, việc xác định chính xác kiểu thảm thực vật rừng (Forest Cover Type) đóng vai trò sống còn đối với công tác quy hoạch lâm nghiệp bền vững:
- **Bảo tồn đa dạng sinh học và quản lý sinh cảnh**: Nhận diện ranh giới phân bố của các quần xã thực vật bản địa, bảo vệ các hành lang sinh thái nhạy cảm (đặc biệt là các vành đai thực vật ven suối) `[4]`.
- **Dự báo và phòng chống cháy rừng**: Mỗi kiểu thảm thực vật có độ nhạy cảm và tốc độ bắt lửa khác biệt hoàn toàn (rừng lá kim giàu nhựa dễ bắt lửa trên sườn dốc khô hạn đối lập với rừng rụng lá và rừng ẩm ven suối) `[2, 5]`.
- **Giảm thiểu chi phí điều tra thực địa**: Thay vì tổ chức các đợt khảo sát trắc địa thủ công tốn kém hàng tỷ đồng tại các vùng núi cao hiểm trở, việc ứng dụng **Trí tuệ nhân tạo (AI)** và **Học máy (Machine Learning)** cho phép tự động hóa quá trình nhận diện kiểu rừng dựa trên dữ liệu mô hình độ cao số (DEM) và bản đồ thổ nhưỡng viễn thám `[2, 3]`.

### 1.2. Bộ dữ liệu nghiên cứu: Forest Covertype (UCI ID 31)
Bộ dữ liệu được thu thập từ 4 khu vực rừng hoang dã thuộc Rừng quốc gia Roosevelt (Roosevelt National Forest), dãy núi Rocky, bang Colorado, Hoa Kỳ `[1, 2]`. Dữ liệu gồm **581.012 dòng quan sát** (mỗi ô lưới $30\text{m} \times 30\text{m}$) và **54 đặc trưng trắc địa** `[1, 3]`:
- **10 biến số liên tục**: Độ cao (`Elevation`: 1.859m – 3.858m), Hướng sườn (`Aspect`: 0° – 360°), Độ dốc (`Slope`: 0° – 66°), Khoảng cách ngang & đứng tới nguồn nước (`Horizontal/Vertical_Distance_To_Hydrology`), Khoảng cách tới đường bộ, 3 chỉ số bóng râm bức xạ mặt trời (`Hillshade` 9am, Noon, 3pm), Khoảng cách tới các điểm cháy rừng lịch sử.
- **4 biến nhị phân Vùng hoang dã (Wilderness Areas)**: Rawah, Neota, Comanche Peak, Cache la Poudre.
- **40 biến nhị phân Loại đất (Soil Types)**: `Soil_Type1` đến `Soil_Type40` (mã hóa One-Hot theo hệ thống Đơn vị đất sinh thái USFS ELU) `[1]`.

Mục tiêu phân loại là **7 kiểu thảm rừng ôn đới núi cao**, phân tầng chặt chẽ theo độ cao và điều kiện thổ nhưỡng `[4, 5]`:
1. **Spruce/Fir** (Vân sam / Linh sam): Đai núi cao cận tuyết Subalpine ($2.800\text{m} - 3.400\text{m}$).
2. **Lodgepole Pine** (Thông Lodgepole): Loài cây tiên phong trên sườn núi trung bình ($2.600\text{m} - 3.100\text{m}$).
3. **Ponderosa Pine** (Thông Ponderosa): Loài chịu hạn sườn núi thấp đón nắng Nam ($1.800\text{m} - 2.600\text{m}$).
4. **Cottonwood/Willow** (Cây bông vải / Liễu): Loài ưa ẩm ven suối thung lũng đáy sườn ($< 2.450\text{m}$).
5. **Aspen** (Bạch dương rung lá): Đất mùn sâu, giàu dinh dưỡng, thoát nước tốt ($2.400\text{m} - 2.900\text{m}$).
6. **Douglas-fir** (Linh sam Douglas): Sườn dốc mát mẻ đai thấp ($2.300\text{m} - 2.800\text{m}$).
7. **Krummholz** (Rừng cây lùn dị dạng): Đai ranh giới băng tuyết (Tree line $> 3.250\text{m} - 3.400\text{m}$).

---

> **Lưu ý về dải độ cao**: các dải trên là *khoảng tham khảo đã làm tròn* tổng hợp từ tài liệu sinh thái học kinh điển `[4, 5]` (**Đã xác minh**: khớp các đai sinh thái của Peet 1981 và Marr 1961), có sự chồng lấn tự nhiên giữa các loài (đặc biệt Spruce/Fir và Lodgepole). Đây là bảng chuẩn duy nhất của bộ tài liệu; các file khác tham chiếu về đây. Ngưỡng phân chia cuối cùng dùng trong các luật logic phải lấy từ dữ liệu thực nghiệm (EDA, Bước 2), không dùng làm ngưỡng cứng suy diễn chủ quan.

## 2. NHỮNG THÁCH THỨC KHOA HỌC VÀ YÊU CẦU ĐỀ TÀI

### 2.1. Thách thức kỹ thuật
1. **Chi phí tính toán trên Google Colab (Colab Resource Constraints)**:
   - Bộ dữ liệu có $581.012$ mẫu $\times$ $54$ thuộc tính (khoảng 0,25 GB nếu lưu `float64`, có thể giảm bằng ép kiểu `float32`/`uint8`), nên việc *nạp dữ liệu* không phải nút thắt. Chi phí chính nằm ở huấn luyện Random Forest nhiều cây sâu và ở việc lặp lại nhiều cấu hình thí nghiệm (ablation, tinh chỉnh siêu tham số). Thời gian/RAM thực tế trên Colab **[CẦN ĐO]**. **[SỐ ĐO]** trên máy cục bộ (24 lõi CPU, 16 GB RAM): chạy toàn bộ notebook ≈ 48 phút; huấn luyện RF 100 cây, không giới hạn độ sâu trên 348.608 mẫu ≈ 21 s; RAM tiến trình tại các điểm kiểm tra ≤ ≈ 2,2 GB.
   - Giải pháp dự kiến: dùng **lấy mẫu phân tầng theo nhãn ($f = 15\%$, $n = 87.152$ mẫu)** cho *pha thử nghiệm nhanh*, giữ nguyên tỷ lệ 7 lớp (xem `CHIEN_LUOC_LAY_MAU_PHAN_TANG.md`). Kích thước mẫu cho mô hình cuối được quyết định bằng **learning curve** (15% / 30% / 50% / 100%), vì giảm dữ liệu có thể làm giảm hiệu năng. **[SỐ ĐO]** E4: Macro-F1 trên D_val tăng từ 0,848 (≈ 87K mẫu) lên 0,921 (348.608 mẫu) và chưa bão hoà, nên mô hình cuối (v2) dùng toàn bộ D_pool.
2. **Mất cân bằng lớp cực đoan (Severe Class Imbalance)** `[1, 3]`:
   - Hai lớp đa số (Lớp 1 & Lớp 2) chiếm tới **$85.22\%$** diện tích rừng ($495.141$ mẫu).
   - Lớp thiểu số (Lớp 4 - *Cottonwood/Willow*) chỉ chiếm vỏn vẹn **$0.473\%$** ($2.747$ mẫu).
   - Nguy cơ: Mô hình ngây thơ chỉ dự đoán Lớp 1 và 2 để đạt Accuracy cao nhưng hoàn toàn thất bại ở Lớp 4 (Accuracy Paradox).
   - Giải pháp dự kiến: kết hợp **lấy mẫu phân tầng** (giữ tỷ lệ lớp) với **trọng số lớp nhạy cảm chi phí (`class_weight='balanced'`)**; tỷ số trọng số $w_4 / w_2 \approx 103:1$ (tính trên tập huấn luyện). Hiệu quả của trọng số là **[GIẢ THUYẾT]**, được kiểm chứng bằng ablation (không trọng số / `balanced` / `balanced_subsample`).
3. **Giả thuyết pha loãng đặc trưng của 40 cột One-Hot đất**:
   - 40 cột đất chiếm $74.07\%$ số chiều. Với $m = 7$ ứng viên mỗi nút, xác suất một nút có $\ge 5$ ứng viên là biến đất khoảng $74.7\%$ (phân phối siêu bội, xem `KY_THUAT_TAO_DAC_TRUNG_SINH_THAI.md`). Điều này *có thể* làm giảm cơ hội của các biến địa hình, nhưng mức ảnh hưởng đến hiệu năng chưa được chứng minh.
   - Giải pháp thử nghiệm: so sánh 3 cách biểu diễn đất — (A) One-Hot nguyên bản; (B) `Soil_Type_Code` (1 cột số nguyên); (C) `Soil_Climatic_Zone` + `Soil_Geologic_Zone` theo USFS ELU — và chọn theo kết quả đo.
4. **Yêu cầu tính giải thích tường minh (Explainable AI - XAI) & "Luật sinh thái rừng"**:
   - Đề tài không chấp nhận mô hình "Hộp đen" (Black-box).
   - Bắt buộc phải **trích xuất được bộ quy tắc logic If-Then tường minh** từ Cây quyết định cắt tỉa, đối chiếu với các quy luật phân tầng sinh thái học trong Peet (1981) `[4]` và Marr (1961) `[5]` (luật mô tả tương quan trong dữ liệu, không phải quan hệ nhân quả).

### 2.2. Yêu cầu triển khai cụ thể
- **Môi trường thực thi**: đề bài (`docs/detai.md`) yêu cầu thực hiện trên **Google Colab**. Hiện trạng: notebook đã chạy trọn vẹn trên máy cục bộ; đường dẫn được viết để chạy được sau khi clone repo lên Colab, nhưng **chưa chạy thử trên Colab** **[CẦN KIỂM TRA]**.
- **Tải dữ liệu tự động**: notebook đọc `data/raw/covtype.data` (hoặc bản `.gz`); nếu không có thì tải bằng `sklearn.datasets.fetch_covtype()`. Bản tải này đã được đối chiếu và trùng khớp từng giá trị, cùng thứ tự dòng với file UCI. Đề bài gợi ý `ucimlrepo` (`fetch_ucirepo(id=31)`); notebook dùng `fetch_covtype()` vì thư viện này đã có sẵn cùng scikit-learn, nên không cần cài thêm. Bản `ucimlrepo` cũng lấy từ UCI 31 nhưng chưa được đối chiếu với file cục bộ.
- **Quy chuẩn mã nguồn**: Viết code sạch chuẩn PEP8, chú thích chi tiết bằng tiếng Việt học thuật.

---

## 3. MỤC ĐÍCH VÀ KẾT QUẢ MONG MUỐN ĐẠT ĐƯỢC

### 3.1. Mục đích cốt lõi
1. **Xây dựng Pipeline học máy tối ưu và chuẩn mực**: Quy trình khép kín từ Nạp dữ liệu $\to$ Phân tích khám phá (EDA) $\to$ Lấy mẫu phân tầng $\to$ Kỹ thuật tạo đặc trưng sinh thái $\to$ Huấn luyện Random Forest nhạy cảm chi phí $\to$ Đánh giá hiệu năng $\to$ Trích xuất luật If-Then sinh thái.
2. **Phân loại chuẩn xác 7 kiểu thảm rừng**: Tối ưu hóa mô hình Random Forest, nâng cao độ chính xác trên toàn bộ 7 loài cây, đặc biệt bảo vệ thành công lớp hiếm Lớp 4.
3. **Mô hình hóa và trích xuất "Luật sinh thái rừng"**: Tự động hóa thuật toán duyệt cây theo chiều sâu (DFS), định lượng chất lượng luật qua $Confidence$, $Support$, $Lift$, và xuất ra bảng "Quy tắc vàng If-Then sinh thái" trực quan, dễ hiểu.

### 3.2. Ngưỡng hiệu năng kỳ vọng (Acceptance Thresholds) — **[KỲ VỌNG, chưa đo]**
> Các ngưỡng dưới đây là mục tiêu tối thiểu do nhóm đặt ra, **không phải kết quả**, và chưa có nguồn trực tiếp xác nhận cho Covertype 7 lớp. Chúng sẽ được thay bằng số đo thực tế ở Bước 5; nếu kết quả cao hơn, báo cáo theo số đo.

- **Độ chính xác tổng thể (Overall Accuracy)**: $\ge 88\%$ trên tập kiểm thử độc lập (mốc tham chiếu từ nghiên cứu gốc: LDA 58.4% và ANN 70.6% — Blackard & Dean `[3]`).
- **Macro F1-score**: $\ge 0.78$.
- **Recall lớp 4 (*Cottonwood/Willow*)**: $\ge 75\%$, báo cáo kèm **khoảng tin cậy** (lớp 4 chỉ có ≈ 82 mẫu trong tập test 20% của mẫu 15%; xem `KE_HOACH_THI_NGHIEM.md`).
- **Thời gian & RAM**: **[CẦN ĐO]** cho từng cấu hình; mục tiêu tham khảo $< 3$ phút và $< 2$ GB cho pha thử nghiệm 15%.

#### Đối chiếu ngưỡng kỳ vọng với số đo — **[SỐ ĐO]**
> Nguồn: notebook mục 5B.6 (D_final: 116.202 mẫu rút từ hold-out cũ, chỉ đánh giá một lần; lớp 4 = 549 mẫu) và mục 5B.7 (E6). KTC 95%: bootstrap 1.000 lần cho Accuracy/Macro-F1, Wilson cho Recall.

| Chỉ số | Ngưỡng kỳ vọng | Bước 4.7 (mẫu 15%) | v2 (Bước 5B) | v2 — vùng chưa thấy (E6) |
| :--- | :---: | :---: | :---: | :---: |
| Accuracy | ≥ 88% | 0,8908 [0,8891–0,8926] | 0,9754 [0,9745–0,9764] | 0,50 (TB 4 vùng) |
| Macro-F1 | ≥ 0,78 | 0,834 [0,829–0,840] | 0,954 [0,951–0,957] | 0,32 (TB 4 vùng, lớp đã thấy) |
| Recall lớp 4 | ≥ 75% | 0,763 [0,726–0,797] | 0,889 [0,860–0,913] | Không đo được: lớp 4 chỉ có ở vùng 4 |

- Cả hai mô hình đạt các ngưỡng kỳ vọng khi đánh giá bằng **chia ngẫu nhiên** trong 4 vùng nghiên cứu.
- Khi kiểm thử trên một vùng hoang dã không có trong dữ liệu huấn luyện, không mô hình nào đạt ngưỡng. Vì vậy các ngưỡng trên chỉ nên được diễn giải cho phạm vi **bên trong khu vực nghiên cứu**; xem hạn chế ở mục 5B.8 của notebook.

### 3.3. Danh mục sản phẩm bàn giao (Deliverables)
1. **Notebook `notebooks/Phan_loai_tham_rung.ipynb`**: tự chứa, chạy từ đầu đến cuối bằng "Run All" (đã kiểm tra trên máy cục bộ). Nút "Open In Colab" và kiểm tra chạy trên Colab: **[CẦN THỰC HIỆN]** ở Bước 8.
2. **Bộ trực quan hóa chuyên sâu**:
   - Heatmap Ma trận nhầm lẫn (Confusion Matrix) gắn nhãn khoa học 7 loài cây.
   - Biểu đồ xếp hạng mức độ quan trọng của đặc trưng địa hình (Feature Importance).
   - Sơ đồ Cây quyết định cắt tỉa trực quan hóa các nhánh rẽ sinh thái.
3. **Bảng "Quy tắc vàng If-Then sinh thái"**: 7–10 quy tắc đại diện tinh hoa kèm chỉ số $Confidence$, $Support$, $Lift$ và phân tích sinh thái thực địa.
4. **Hàm dự đoán tương tác thực địa**: Module Python tương tác cho phép nhập tọa độ/địa hình và xuất kết luận kiểu rừng kèm luật kích hoạt.
5. **Hệ thống tài liệu hoàn chỉnh trong thư mục `docs/`**.
6. **Báo cáo thí nghiệm ablation và learning curve** (bảng so sánh các cấu hình, khoảng tin cậy; xem `KE_HOACH_THI_NGHIEM.md`).

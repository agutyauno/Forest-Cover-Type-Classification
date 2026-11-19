# NGHIÊN CỨU LÝ THUYẾT VÀ PHƯƠNG PHÁP LUẬN: LẤY MẪU PHÂN TẦNG VÀ HỌC NHẠY CẢM CHI PHÍ TRÊN DỮ LIỆU ĐỊA HÌNH QUY MÔ LỚN
## Theoretical Foundations of Stratified Sampling and Cost-Sensitive Learning on Large-Scale Geospatial Imbalanced Data

> **Tài liệu tham chiếu liên quan**:
> - [Bối cảnh, Yêu cầu và Mục tiêu đề tài](../01_tong_quan_yeu_cau/BOI_CANH_YEU_CAU_MUC_TIEU.md)
> - [Kế hoạch triển khai kỹ thuật](../02_ke_hoach_va_kien_truc/KE_HOACH_TRIEN_KHAI.md)
> - [Cơ sở lý thuyết thuật toán Random Forest](THUAT_TOAN_RANDOM_FOREST.md)
> - [Danh mục tài liệu tham khảo & Dẫn chứng khoa học](../04_tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md)

---

> **Trạng thái**: thiết kế trước khi thực hiện. Các nhận định về hiệu năng là **[GIẢ THUYẾT]** cần kiểm chứng bằng thí nghiệm (`KE_HOACH_THI_NGHIEM.md`).

## 1. ĐẶT VẤN ĐỀ VÀ CƠ SỞ KHOA HỌC

Trong phân tích dữ liệu không gian và mô hình hóa sinh thái học cảnh quan, các tập dữ liệu trắc địa viễn thám thường mang hai đặc tính cố hữu: **quy mô quan sát lớn (Big Data Scale)** và **phân bố bất đối xứng cực đoan giữa các lớp phân loại (Severe Class Imbalance)** `[1, 3]`.

Tại địa bàn Rừng quốc gia Roosevelt (Roosevelt National Forest) (Colorado, Hoa Kỳ), bộ dữ liệu Forest Covertype (UCI ID 31) chứa $N = 581.012$ quan sát địa hình số `[1]`. Tuy nhiên, sự phân bố sinh thái thực địa không tuân theo phân phối đều mà phản ánh tính chất áp đảo của các sinh cảnh chiếm ưu thế:
- Nhóm sinh cảnh diện rộng (Lớp 1 - Vân sam và Lớp 2 - Thông Lodgepole) chiếm $85.22\%$ tổng diện tích điều tra `[1, 3]`.
- Nhóm sinh cảnh hẹp ven thủy vực (Lớp 4 - Cây bông vải / Liễu ven suối *Cottonwood/Willow*) chỉ xuất hiện với tần suất $0.47\%$ ($2.747$ mẫu) `[1, 4]`.

Khi huấn luyện mô hình tập hợp (nhiều cây sâu) trên toàn bộ $581.012$ mẫu, chi phí chủ yếu nằm ở thời gian huấn luyện và ở việc lặp lại nhiều cấu hình thí nghiệm (nạp dữ liệu chỉ khoảng 0,25 GB ở `float64`). Mức tiêu thụ thực tế trên Google Colab **[CẦN ĐO]**. Vì vậy dự án dùng mẫu con 15% cho *pha thử nghiệm nhanh*, và quyết định kích thước mẫu cuối bằng learning curve.

Do đó, việc thiết lập một **chiến lược lấy mẫu có cơ sở toán học vững chắc** nhằm giảm thiểu kích thước không gian mẫu nhưng vẫn bảo toàn cấu trúc phân phối nhãn là một lựa chọn hợp lý cho pha thử nghiệm `[13]`; mức đánh đổi về hiệu năng được đo bằng learning curve.

---

## 2. MÔ HÌNH TOÁN HỌC CỦA PHƯƠNG PHÁP LẤY MẪU PHÂN TẦNG (STRATIFIED SAMPLING)

### 2.1. Định nghĩa không gian xác suất
Xét tổng thể thống kê hữu hạn $\mathcal{U} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$, trong đó $\mathbf{x}_i \in \mathbb{R}^d$ là vector đặc trưng địa hình ($d = 54$) và $y_i \in \mathcal{Y} = \{1, 2, \dots, K\}$ là nhãn kiểu thảm rừng mục tiêu ($K = 7$).

Tổng thể $\mathcal{U}$ được phân hoạch thành $K$ tập con rời rạc (gọi là các tầng - Strata) $\mathcal{U}_1, \mathcal{U}_2, \dots, \mathcal{U}_K$ thỏa mãn:

$$\mathcal{U} = \bigcup_{k=1}^{K} \mathcal{U}_k \quad \text{và} \quad \mathcal{U}_i \cap \mathcal{U}_j = \emptyset, \quad \forall i \ne j$$

Số lượng phần tử của tầng thứ $k$ ký hiệu là $N_k = |\mathcal{U}_k|$, thỏa mãn $\sum_{k=1}^{K} N_k = N$.

Xác suất tiên nghiệm khách quan của lớp $k$ trong tự nhiên được xác định bởi:

$$p_k = P(y = k) = \frac{N_k}{N}$$

### 2.2. Phân bổ mẫu theo tỷ lệ (Proportional Allocation)
Phương pháp lấy mẫu phân tầng theo tỷ lệ (Proportional Stratified Sampling) thiết lập quy tắc trích xuất từ mỗi tầng $\mathcal{U}_k$ một tập con ngẫu nhiên đơn giản không hoàn lại $\mathcal{S}_k \subset \mathcal{U}_k$ với kích thước $n_k = |\mathcal{S}_k|$.

Với quy mô tập mẫu mục tiêu là $n$ ($n < N$), kích thước mẫu của từng tầng được xác định theo nguyên lý phân bổ tỷ lệ thuận `[13]`:

$$n_k = n \times \frac{N_k}{N} = n \times p_k$$

Tỷ lệ lấy mẫu chung (Sampling Fraction) cho toàn bộ các tầng là hằng số:

$$f = \frac{n_k}{N_k} = \frac{n}{N}, \quad \forall k \in \{1, \dots, K\}$$

Trong nghiên cứu này, hệ số lấy mẫu được xác định là $f = 0.15$ ($15.0\%$), tương ứng với quy mô tập mẫu đại diện $n = 87.152$ quan sát.

---

## 3. TÍNH CHẤT PHƯƠNG SAI CỦA LẤY MẪU PHÂN TẦNG (ƯỚC LƯỢNG TRUNG BÌNH)

> **Phạm vi áp dụng**: định lý dưới đây là kết quả của lý thuyết chọn mẫu về *phương sai của ước lượng trung bình* $\bar{y}$ của một biến $Y$. Nó biện minh cho việc phân tầng theo nhãn để *ước lượng tỷ lệ các lớp ổn định hơn*. Nó **không** chứng minh rằng mô hình phân loại huấn luyện trên mẫu phân tầng sẽ có accuracy/F1 cao hơn; điều đó cần được đo bằng thí nghiệm.

Theo lý thuyết chọn mẫu cổ điển của Cochran (1977) `[13]`, ưu thế cốt lõi của Lấy mẫu phân tầng so với Lấy mẫu ngẫu nhiên đơn giản (Simple Random Sampling - SRS) nằm ở khả năng triệt tiêu phương sai giữa các tầng (Between-strata Variance).

Xét một hàm chỉ thị phân loại hoặc một biến ngẫu nhiên đo lường $Y$. Ước lượng không chệch của giá trị trung bình tổng thể $\bar{Y}$ qua mẫu phân tầng là:

$$\bar{y}_{\text{st}} = \sum_{k=1}^{K} W_k \bar{y}_k, \quad \text{với } W_k = \frac{N_k}{N}$$

Phương sai của ước lượng phân tầng tỷ lệ được chứng minh bởi công thức Cochran `[13]`:

$$\text{Var}(\bar{y}_{\text{st}}) = \frac{1 - f}{n} \sum_{k=1}^{K} W_k S_k^2$$

Trong khi đó, phương sai của ước lượng qua lấy mẫu ngẫu nhiên đơn giản cùng kích thước $n$ là:

$$\text{Var}(\bar{y}_{\text{srs}}) \approx \frac{1 - f}{n} S^2 = \frac{1 - f}{n} \left[ \sum_{k=1}^{K} W_k S_k^2 + \sum_{k=1}^{K} W_k (\bar{Y}_k - \bar{Y})^2 \right]$$

Từ hai đẳng thức trên, sai khác phương sai giữa hai phương pháp là:

$$\text{Var}(\bar{y}_{\text{srs}}) - \text{Var}(\bar{y}_{\text{st}}) = \frac{1 - f}{n} \sum_{k=1}^{K} W_k (\bar{Y}_k - \bar{Y})^2 \ge 0$$

> **Hệ quả định lý**:
> Vì đại lượng $\sum_{k=1}^{K} W_k (\bar{Y}_k - \bar{Y})^2$ đại diện cho phương sai giữa các tầng (đo lường sự khác biệt về mặt sinh thái và địa hình giữa các loài cây), giá trị này luôn không âm ($\ge 0$). Do đó, **phương sai sai số chọn mẫu của Lấy mẫu phân tầng luôn luôn nhỏ hơn hoặc bằng Lấy mẫu ngẫu nhiên đơn giản**:
> $$\text{Var}(\bar{y}_{\text{st}}) \le \text{Var}(\bar{y}_{\text{srs}})$$
> Đẳng thức chỉ xảy ra khi trung bình của mọi tầng hoàn toàn đồng nhất ($\bar{Y}_1 = \bar{Y}_2 = \dots = \bar{Y}_K$). Trong bài toán sinh thái rừng với sự phân tầng độ cao rõ rệt `[4]`, chênh lệch giữa các tầng có thể lớn, cho thấy phân tầng theo nhãn làm ổn định ước lượng tỷ lệ các lớp (trong phạm vi áp dụng nêu trên).

---

## 4. BẢNG PHÂN BỔ THỰC NGHIỆM TRÊN BỘ DỮ LIỆU FOREST COVERTYPE

Áp dụng mô hình phân bổ tỷ lệ với $f = 0.15$ trên tập dữ liệu $581.012$ mẫu `[1]`, cấu trúc tập dữ liệu nghiên cứu được định lượng chi tiết trong bảng sau:

| Mã lớp ($k$) | Danh pháp sinh thái / Kiểu rừng `[1, 4]` | Quy mô tổng thể ($N_k$) `[1]` | Tỷ trọng tự nhiên ($p_k$) | Số mẫu phân tầng trích xuất ($n_k$) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Vân sam / Linh sam (*Spruce/Fir*) | $211.840$ | $36.460\%$ | **$31.776$** |
| **2** | Thông Lodgepole (*Lodgepole Pine*) | $283.301$ | $48.759\%$ | **$42.495$** |
| **3** | Thông Ponderosa (*Ponderosa Pine*) | $35.754$ | $6.154\%$ | **$5.363$** |
| **4** | Cây bông vải / Liễu (*Cottonwood/Willow*) | $2.747$ | **$0.473\%$** | **$412$** |
| **5** | Bạch dương rung lá (*Aspen*) | $9.493$ | $1.634\%$ | **$1.424$** |
| **6** | Linh sam Douglas (*Douglas-fir*) | $17.367$ | $2.989\%$ | **$2.605$** |
| **7** | Rừng lùn cận tuyết (*Krummholz*) | $20.510$ | $3.531\%$ | **$3.077$** |
| **$\Sigma$** | **Toàn bộ 7 kiểu thảm thực vật** | **$581.012$** | **$100.000\%$** | **$87.152$** |

### Hệ quả thực tế của kích thước mẫu $n = 87.152$:
1. **Số mẫu mỗi lớp là cố định** theo phân bổ tỷ lệ (ví dụ lớp 4: $n_4 = 412$); đây không phải biến ngẫu nhiên nên không áp dụng xấp xỉ Poisson/nhị thức cho "xác suất thiếu mẫu". Các con số trong bảng được làm tròn sao cho tổng bằng $87.152$; `train_test_split(stratify=y)` có thể lệch ±1 mẫu mỗi lớp.
2. **Lớp 4 sau chia 80/20** còn khoảng $330$ mẫu huấn luyện và $82$ mẫu kiểm thử. Recall đo trên $82$ mẫu có độ bất định lớn (ví dụ Recall $\approx 0.80$ có khoảng tin cậy 95% xấp xỉ $\pm 9$ điểm phần trăm). Vì vậy cần báo cáo khoảng tin cậy (Wilson hoặc bootstrap), lặp nhiều seed hoặc CV phân tầng, và đánh giá thêm trên hold-out lớn (≈ $2.335$ mẫu lớp 4).
3. **Kích thước 15% chỉ là lựa chọn cho pha thử nghiệm**; tính đủ của nó được kiểm tra bằng learning curve, không suy ra từ quy tắc kích thước mẫu tiệm cận.

---

## 5. COST-SENSITIVE LEARNING VÀ CƠ CHẾ BÙ TRỪ

Mặc dù Lấy mẫu phân tầng bảo toàn trung thực phân phối tiên nghiệm của tự nhiên, tỷ lệ bất đối xứng giữa lớp cực đại ($N_2 = 283.301$) và lớp cực tiểu ($N_4 = 2.747$) đạt mức $\approx 103:1$. 

Nếu tối ưu hóa hàm mất mát tiêu chuẩn (như hàm độ vẩn đục Gini hoặc hàm mất mát chéo Entropy không trọng số), thuật toán quy nạp cây sẽ bị chi phối bởi lớp đa số nhằm tối đa hóa độ chính xác tổng thể (Accuracy Paradox) `[7]`.

Để giải quyết mâu thuẫn này, nghiên cứu tích hợp cơ chế **Trọng số phân loại nhạy cảm chi phí (Cost-Sensitive Class Weighting)** vào không gian phân nhánh của cây quyết định.

### 5.1. Công thức trọng số nghịch đảo tần suất
Hệ thống trọng số phạt $w_k$ cho lớp thứ $k$ được tính theo heuristic `'balanced'` của Scikit-learn `[14]`:

$$w_k = \frac{n}{K \times n_k}$$

Trong đó:
- $n$: Tổng số quan sát **trong tập huấn luyện** ($n \approx 69.721$ sau khi chia 80/20 từ mẫu 87.152).
- $K$: Số lớp phân loại ($K = 7$).
- $n_k$: Số lượng quan sát thực tế thuộc lớp $k$.

### 5.2. Tác động của trọng số lên hàm mục tiêu Gini
Tại một nút phân nhánh $t$ bất kỳ của cây quyết định, độ vẩn đục Gini có trọng số được định nghĩa lại:

$$Gini_w(t) = 1 - \sum_{k=1}^{K} \left( \frac{\sum_{i \in t, y_i=k} w_k}{\sum_{i \in t} w_{y_i}} \right)^2$$

Ví dụ trên tập huấn luyện ($n \approx 69.721$; lớp 2: $n_2 \approx 33.996$; lớp 4: $n_4 \approx 330$):
- $w_2 = \frac{69.721}{7 \times 33.996} \approx \mathbf{0.293}$
- $w_4 = \frac{69.721}{7 \times 330} \approx \mathbf{30.2}$

Tỷ số trọng số giữa Lớp 4 và Lớp 2 là $\frac{w_4}{w_2} = \frac{n_2}{n_4} \approx 103$.

> **Ý nghĩa phương pháp luận (kỳ vọng, cần kiểm chứng)**:
> Trong hàm mục tiêu có trọng số, một mẫu Lớp 4 (*Cottonwood/Willow*) có trọng số gấp ≈ $103$ lần một mẫu Lớp 2 (*Lodgepole Pine*). Kỳ vọng điều này khiến cây ưu tiên tìm biên phân tách cho Lớp 4, tăng **Recall** và **Macro F1-score**; nhưng trọng số cũng có thể làm giảm Precision lớp hiếm, và với Random Forest lợi ích đôi khi nhỏ. Do đó phải so sánh `class_weight=None`, `'balanced'`, `'balanced_subsample'` trong ablation.

---

## 6. SO SÁNH CÁC PHƯƠNG ÁN XỬ LÝ MẤT CÂN BẰNG (METHODOLOGICAL COMPARISON)

| Tiêu chí | Phân tầng + Trọng số lớp (đề xuất) | Lấy mẫu ngẫu nhiên đơn giản (SRS) | Random Undersampling | SMOTE |
| :--- | :--- | :--- | :--- | :--- |
| **Bảo toàn tỷ lệ lớp ($P(Y)$)** | Có (sai khác do làm tròn) | Dao động ngẫu nhiên; lớp hiếm có thể lệch | Không (cân bằng ≈ 1:1) | Không (bóp méo phân phối) |
| **Thông tin bị bỏ** | Bỏ 85% dữ liệu (pha thử nghiệm) | Bỏ 85% dữ liệu | Rất lớn (giữ ≈ $n_4$ mẫu mỗi lớp) | Không bỏ mẫu gốc |
| **Mẫu phi thực tế** | Không | Không | Không | Có nguy cơ: nội suy với biến nhị phân/phân loại (đất, khu vực) có thể tạo tổ hợp không có thật |
| **Chi phí tính toán** | **[CẦN ĐO]** | **[CẦN ĐO]** | Thấp (ít mẫu) | Tăng thêm bước k-NN và số mẫu |
| **Hiệu năng lớp hiếm** | **[GIẢ THUYẾT]** | **[GIẢ THUYẾT]** | **[GIẢ THUYẾT]** | **[GIẢ THUYẾT]** |

Chưa có số đo cho các ô hiệu năng; chúng sẽ được điền từ ablation. Các phương án nên đưa vào so sánh thêm: `class_weight='balanced_subsample'`, `BalancedRandomForest` (thư viện imbalanced-learn), điều chỉnh ngưỡng xác suất; SMOTE gốc xem Chawla và cộng sự (2002) `[17]`.

---

## 7. KẾT LUẬN

Kết hợp **lấy mẫu phân tầng theo tỷ lệ ($f = 15\%$)** với **trọng số lớp ($w_k$)** là phương án thiết kế hợp lý cho pha thử nghiệm: giữ nguyên tỷ lệ 7 lớp và đưa thông tin lớp hiếm vào hàm mục tiêu. Tuy nhiên, hiệu quả thực tế (so với không trọng số, dùng nhiều dữ liệu hơn, hay các phương án khác) là **giả thuyết** và sẽ được kiểm chứng bằng ablation, learning curve và đánh giá trên hold-out lớn.

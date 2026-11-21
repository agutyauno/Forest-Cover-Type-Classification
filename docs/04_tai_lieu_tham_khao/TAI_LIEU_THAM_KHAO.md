# DANH MỤC TÀI LIỆU THAM KHẢO VÀ NGUỒN DẪN CHỨNG KHOA HỌC
## Đề tài: Phân loại kiểu thảm rừng từ dữ liệu địa hình (Forest Cover Type Classification)

Tài liệu này tổng hợp toàn bộ các nguồn dẫn chứng khoa học, bài báo gốc, tài liệu trắc địa lâm nghiệp và các nghiên cứu thực nghiệm chuẩn (benchmarks) làm căn cứ cho các số liệu, ngưỡng sinh thái và chỉ số mục tiêu được sử dụng trong dự án.

> **Trạng thái tài liệu**: Toàn bộ các nguồn trích dẫn `[1]`–`[17]`, tác giả, tiêu đề, mã DOI (đặc biệt `[4]`), các dải cao độ sinh thái Peet/Marr, và bảng ánh xạ mã đất USFS ELU so với `covtype.info` đã được **đối chiếu và xác minh trực tiếp với văn liệu gốc**. Cột *Trạng thái* ở Bảng 1 phản ánh kết quả đối chiếu này.

---

## 1. BẢNG ĐỐI CHIẾU SỐ LIỆU VÀ NGUỒN TRÍCH DẪN

| Số liệu / Nội dung sử dụng trong đề tài | Giá trị cụ thể | Nguồn | Trạng thái |
| :--- | :--- | :--- | :--- |
| **Quy mô tập dữ liệu Covertype** | 581.012 dòng, 54 đặc trưng, 7 kiểu thảm rừng | UCI `[1]`, Blackard (1998) `[2]` | Khớp thống kê công bố; xác nhận lại bằng code ở Bước 1 |
| **Hai lớp đa số (Lớp 1 & 2)** | 495.141 mẫu (211.840 + 283.301; ≈ 85.22%) | `[1]`, `[3]` | Như trên |
| **Lớp thiểu số (Lớp 4 - Cottonwood/Willow)** | 2.747 mẫu (≈ 0.473%) | `[1]`, `[3]` | Như trên |
| **Dải độ cao các loài** (Cottonwood < 2.450m; Ponderosa 1.800–2.600m; Douglas-fir 2.300–2.800m; Spruce/Fir 2.800–3.400m; Krummholz > 3.250m) | Bảng chuẩn ở `BOI_CANH_YEU_CAU_MUC_TIEU.md` | Peet (1981) `[4]`, Marr (1961) `[5]` | **Đã xác minh**: Khớp các đai sinh thái chuẩn của Peet & Marr; số làm tròn, có chồng lấn do vi khí hậu/hướng phơi; ngưỡng phân định cuối cùng lấy từ EDA/dữ liệu |
| **Xác suất mẫu ngoài túi (OOB) ≈ 36.8%** | $\lim_{N \to \infty} (1 - 1/N)^N = 1/e \approx 0.368$ | Breiman (1996) `[6]`, (2001) `[7]` | Kết quả toán học chuẩn |
| **Số đặc trưng ngẫu nhiên mỗi nút** | $m = \lfloor\sqrt{p}\rfloor = 7$ với $p = 54$ | Breiman (2001) `[7]`, Ho (1998) `[8]`, Scikit-learn `[14]` | Mặc định phổ biến (heuristic), không phải giá trị tối ưu đã chứng minh |
| **Ngưỡng accuracy mục tiêu** | ≥ 88% (Macro F1 ≥ 0.78; Recall lớp 4 ≥ 75%) | Do nhóm đặt; `[9]` chỉ hỗ trợ nhận định chung | **[KỲ VỌNG]**, chưa có nguồn trực tiếp cho Covertype 7 lớp; sẽ thay bằng số đo |
| **Baseline truyền thống** | LDA: 58.38%; ANN: 70.58% | Blackard & Dean (1999) `[3]`, `covtype.info` | **Đã xác minh**: Khớp chính xác kết quả kiểm thử trên 565.892 mẫu trong bài báo gốc `[3]` và tài liệu `covtype.info` |
| **Trích xuất luật từ mô hình** | Cây xấp xỉ mô hình, dùng chính mô hình làm "oracle" gán nhãn | Craven & Shavlik (1996) `[11]`, Breiman et al. (1984) `[12]` | Phương pháp luận; dự án áp dụng bằng cách huấn luyện cây trên dự đoán của RF |
| **Phương sai của lấy mẫu phân tầng** | $\text{Var}(\bar{y}_{\text{st}}) \le \text{Var}(\bar{y}_{\text{srs}})$ (ước lượng trung bình) | Cochran (1977) `[13]` | Đúng cho ước lượng trung bình; không suy ra hiệu năng mô hình |
| **Trọng số lớp `'balanced'`** | $w_k = n / (K \cdot n_k)$ | Scikit-learn `[14]` | Heuristic của thư viện |
| **Thiên lệch của Gini importance; tự tương quan không gian** | Dùng Permutation Importance; cảnh báo kết quả lạc quan khi chia ngẫu nhiên | Strobl et al. (2007) `[15]`; Roberts et al. (2017) `[16]` | Bổ sung sau rà soát |

---

## 2. DANH MỤC TÀI LIỆU THAM KHẢO CHI TIẾT

### Nhóm 1: Bài báo công bố dữ liệu gốc và nghiên cứu địa bàn Rừng quốc gia Roosevelt (Roosevelt National Forest)
- **`[1]` UCI Machine Learning Repository (1998)**:
  - *Tác giả*: Blackard, J. A., Dean, D. J., & Anderson, C. W. (**Đã xác minh**: khớp chính xác danh sách Creators/Donors trên trang UCI và `covtype.info`).
  - *Tên bộ dữ liệu*: *Covertype Dataset (ID: 31)*.
  - *Đường dẫn*: [https://archive.ics.uci.edu/dataset/31/covertype](https://archive.ics.uci.edu/dataset/31/covertype)
  - *DOI*: `10.24432/C50W2D`.
  - *Mô tả*: Cung cấp định dạng dữ liệu, định nghĩa 54 thuộc tính (10 số, 4 vùng hoang dã, 40 loại đất USFS ELU) và thống kê tần số 7 lớp.

- **`[2]` Luận án Tiến sĩ công trình gốc (1998)**:
  - *Tác giả*: Blackard, Jock A.
  - *Tiêu đề*: *Comparison of Neural Networks and Discriminant Analysis in Predicting Forest Cover Types* (**Đã xác minh**: tiêu đề nguyên văn theo hồ sơ công trình của Colorado State University và mục 3 trong `covtype.info`, 165 trang).
  - *Đơn vị*: Department of Forest Sciences, Colorado State University, Fort Collins, Colorado.
  - *Mô tả*: Luận án tiến sĩ chi tiết mô tả quy trình thu thập dữ liệu viễn thám DEM, bản đồ thổ nhưỡng GIS của Cục Kiểm lâm Hoa Kỳ (USFS) tại Rừng quốc gia Roosevelt (Roosevelt National Forest).

- **`[3]` Bài báo công bố kết quả quốc tế (1999)**:
  - *Tác giả*: Blackard, J. A., & Dean, D. J.
  - *Tiêu đề*: *"Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables"*.
  - *Tạp chí*: *Computers and Electronics in Agriculture*, Tập 24, Số 3, tr. 131–151.
  - *DOI*: `10.1016/S0168-1699(99)00046-0`.
  - *Kết quả công bố*: Mô hình phân tích biệt số tuyến tính (Linear Discriminant Analysis) đạt độ chính xác 58.38%; Mô hình mạng nơ-ron truyền thẳng (Feedforward MLP ANN) cuối cùng đạt độ chính xác 70.58%; 30 mạng khởi tạo trọng số ngẫu nhiên có độ chính xác trung bình 70.52% (khoảng tin cậy 95%: 70.26%–70.80%).

---

### Nhóm 2: Nghiên cứu thực địa về Sinh thái học thực vật dãy núi Rocky (Colorado Front Range)
- **`[4]` Nghiên cứu phân tầng thực vật Front Range (1981)**:
  - *Tác giả*: Peet, Robert K.
  - *Tiêu đề*: *"Forest vegetation of the Colorado Front Range: Composition and dynamics"*.
  - *Tạp chí*: *Vegetatio*, Tập 45, Số 1, tr. 3–75.
  - *DOI*: `10.1007/BF00240202` (**Đã xác minh và hiệu đính**: sửa từ mã lỗi cũ `10.1007/BF00119221`).
  - *Ý nghĩa sinh thái*: Công trình kinh điển về các đai thực vật ở Colorado Front Range. Các dải độ cao dưới đây đã được đối chiếu khớp với các đai sinh thái của Peet (1981) và Marr (1961); bảng chuẩn dùng cho dự án nằm ở `BOI_CANH_YEU_CAU_MUC_TIEU.md`:
    - *Đai chân núi / ven suối (Riparian corridor)* (< 2.400m / 2.450m): Nơi quần tụ của cây lá rộng ưa ẩm *Populus angustifolia* (Cottonwood) và *Salix* (Willow).
    - *Đai rừng núi thấp (Lower Montane)* (1.800m – 2.600m): Độc tôn của Thông Ponderosa (*Pinus ponderosa*) trên các sườn dốc khô hạn hứng nắng Nam.
    - *Đai rừng núi cao (Upper Montane)* (2.300m – 2.900m): Khu vực mát mẻ của Linh sam Douglas (*Pseudotsuga menziesii*) và Bạch dương (*Populus tremuloides*).
    - *Đai cận tuyết (Subalpine)* (2.800m – 3.400m): Rừng rậm Vân sam (*Picea engelmannii*), Linh sam (*Abies lasiocarpa*) và Thông Lodgepole (*Pinus contorta*).
    - *Ranh giới cây gỗ (Alpine Timberline / Krummholz)* (> 3.250m – 3.400m): Hiện tượng gió rét cực độ uốn cong cây thành bụi lùn rạp sát đất đá — quần xã *Krummholz*.

- **`[5]` Hệ sinh thái sườn Đông rặng Front Range (1961)**:
  - *Tác giả*: Marr, John W.
  - *Tiêu đề*: *Ecosystems of the East Slope of the Front Range in Colorado*.
  - *Nhà xuất bản*: University of Colorado Studies, Series in Biology No. 8, Boulder, Colorado.
  - *Mô tả*: Cung cấp phân chia đai sinh thái chuẩn (Montane 5.600–9.000 ft / 1.700–2.750m, Subalpine 9.000–11.500 ft / 2.750–3.500m, Alpine > 11.500 ft / 3.500m, Timberline Krummholz 3.300–3.500m), bản đồ vi khí hậu và hướng đón nắng (Aspect) ảnh hưởng đến sự phân bố của 7 loài cây bản địa.

---

### Nhóm 3: Cơ sở lý thuyết Thuật toán Học máy, Thống kê và Chọn mẫu
- **`[6]` Nền tảng Bootstrap Aggregating - Bagging (1996)**:
  - *Tác giả*: Breiman, Leo.
  - *Tiêu đề*: *"Bagging Predictors"*.
  - *Tạp chí*: *Machine Learning*, Tập 24, Số 2, tr. 123–140.
  - *Nội dung*: Chứng minh toán học về việc kết hợp các mô hình bất ổn định (unstable models) thông qua lấy mẫu có hoàn lại giúp giảm phương sai của bộ dự đoán; độ chệch về cơ bản tương đương mô hình đơn lẻ.

- **`[7]` Thuật toán Random Forests chính thức (2001)**:
  - *Tác giả*: Breiman, Leo.
  - *Tiêu đề*: *"Random Forests"*.
  - *Tạp chí*: *Machine Learning*, Tập 45, Số 1, tr. 5–32.
  - *DOI*: `10.1023/A:1010933404324`.
  - *Nội dung*: Công bố thuật toán Random Forest, cơ chế chọn ngẫu nhiên một tập con $m$ đặc trưng tại mỗi nút, ước lượng lỗi ngoài túi Out-Of-Bag (OOB) và phép đo Gini Feature Importance.

- **`[8]` Phương pháp không gian con ngẫu nhiên (1998)**:
  - *Tác giả*: Ho, Tin Kam.
  - *Tiêu đề*: *"The random subspace method for constructing decision forests"*.
  - *Tạp chí*: *IEEE Transactions on Pattern Analysis and Machine Intelligence*, Tập 20, Số 8, tr. 832–844.

- **`[9]` So sánh thực nghiệm quy mô lớn các thuật toán Học máy (2006)**:
  - *Tác giả*: Caruana, Rich, & Niculescu-Mizil, Alexandru.
  - *Tiêu đề*: *"An empirical comparison of supervised learning algorithms"*.
  - *Hội nghị*: *Proceedings of the 23rd International Conference on Machine Learning (ICML '06)*, tr. 161–168.
  - *DOI*: `10.1145/1143844.1143865`.
  - *Nội dung*: So sánh thực nghiệm 10 thuật toán học có giám sát trên các bài toán dữ liệu dạng bảng.
  - **Đã xác minh và làm rõ phạm vi**: Nghiên cứu này đánh giá trên bài toán **phân loại nhị phân (binary)**, trong đó tập `COVTYPE` được chuyển đổi thành bài toán 1-vs-rest trên 4.000 mẫu huấn luyện. Bài báo **hoàn toàn KHÔNG cung cấp con số accuracy 88–91% trên bài toán Covertype 7 lớp đa biến**. Vì vậy, tài liệu này chỉ dùng làm luận cứ phương pháp luận (chứng minh Random Forest và Gradient Boosted Trees thuộc nhóm mô hình mạnh nhất trên dữ liệu bảng), không làm căn cứ cho số liệu KPI cụ thể của đề tài.

- **`[10]` Cuộc thi Kaggle Forest Cover Type Prediction (2014)**:
  - *Đường dẫn chính thức*: [https://www.kaggle.com/c/forest-cover-type-prediction](https://www.kaggle.com/c/forest-cover-type-prediction)
  - *Đặc điểm giao thức Kaggle*: Cuộc thi sử dụng tập huấn luyện giới hạn gồm **15.120 mẫu** (cân bằng 2.160 mẫu/lớp) và đánh giá trên tập kiểm thử **565.892 mẫu** (toàn bộ phần còn lại). Trên giao thức khắt khe này, các giải pháp Random Forest cơ bản đạt ~73–75%, còn các mô hình tối ưu hóa tập hợp (Extra Trees, XGBoost, ensembling) đạt cao nhất khoảng **85%–86.6%**.
  - *Phân biệt với thí nghiệm phân chia ngẫu nhiên*: Các notebook cộng đồng đạt 88%–93%+ là do thực hiện chia ngẫu nhiên (Random 70/30 hoặc 80/20) trên toàn bộ 581.012 dòng dữ liệu, nơi 2 lớp đa số chiếm tới 85.22% và hiện tượng tự tương quan không gian giúp cải thiện điểm số. Mục này được giữ lại dưới dạng **tham chiếu thực tiễn cộng đồng** với đầy đủ URL và ghi chú giao thức, không dùng làm baseline học thuật chuẩn (baseline học thuật chuẩn của đề tài là Blackard & Dean `[3]`).

- **`[11]` Trích xuất tri thức từ mô hình học máy (1996)**:
  - *Tác giả*: Craven, Mark W., & Shavlik, Jude W.
  - *Tiêu đề*: *"Extracting tree-structured representations of trained networks"*.
  - *Kỷ yếu*: *Advances in Neural Information Processing Systems (NeurIPS '96)*, Tập 8, tr. 24–30.
  - *Nội dung*: Phương pháp trích xuất cây quyết định xấp xỉ một mô hình phức tạp, dùng chính mô hình đó làm "oracle" để gán nhãn dữ liệu huấn luyện cho cây (mô hình đại diện/surrogate).

- **`[12]` Lý thuyết Cây phân loại và Hồi quy CART (1984)**:
  - *Tác giả*: Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J.
  - *Sách chuyên khảo*: *Classification and Regression Trees*.
  - *Nhà xuất bản*: Wadsworth & Brooks/Cole Advanced Books & Software, Monterey, California.
  - *Nội dung*: Nền tảng thuật toán CART, công thức độ vẩn đục Gini Impurity và kỹ thuật tỉa cành theo độ phức tạp chi phí (Cost-Complexity Pruning).

- **`[13]` Lý thuyết Kỹ thuật chọn mẫu cổ điển (1977)**:
  - *Tác giả*: Cochran, William G.
  - *Sách chuyên khảo*: *Sampling Techniques* (Tái bản lần 3).
  - *Nhà xuất bản*: John Wiley & Sons, New York.
  - *Nội dung*: Giáo trình kinh điển đặt nền móng toán học cho phương pháp Lấy mẫu phân tầng (Stratified Sampling), nguyên lý phân bổ tỷ lệ thuận (Proportional Allocation) và chứng minh định lý phương sai tối thiểu $\text{Var}(\bar{y}_{\text{st}}) \le \text{Var}(\bar{y}_{\text{srs}})$.

---

### Nhóm 4: Bổ sung sau rà soát
- **`[14]` Scikit-learn**: Pedregosa, F. và cộng sự (2011). *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research*, 12, 2825–2830. Dùng cho: cài đặt `RandomForestClassifier`, `DecisionTreeClassifier`, heuristic `class_weight='balanced'`, `permutation_importance`.
- **`[15]` Thiên lệch của độ quan trọng đặc trưng trong Random Forest**: Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). *Bias in random forest variable importance measures: Illustrations, sources and a solution*. *BMC Bioinformatics*, 8, 25.
- **`[16]` Kiểm định chéo cho dữ liệu có cấu trúc không gian**: Roberts, D. R. và cộng sự (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. *Ecography*, 40(8), 913–929.
- **`[17]` SMOTE**: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. *Journal of Artificial Intelligence Research*, 16, 321–357.

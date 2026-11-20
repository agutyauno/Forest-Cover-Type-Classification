# PHƯƠNG PHÁP TRÍCH XUẤT LUẬT SINH THÁI RỪNG (RULE EXTRACTION METHODOLOGY)
## Đề tài: Phân loại kiểu thảm rừng từ dữ liệu địa hình (Forest Cover Type Classification)

> **Tài liệu tham chiếu liên quan**:
> - [Bối cảnh, Yêu cầu và Mục tiêu đề tài](../01_tong_quan_yeu_cau/BOI_CANH_YEU_CAU_MUC_TIEU.md)
> - [Kế hoạch triển khai kỹ thuật](../02_ke_hoach_va_kien_truc/KE_HOACH_TRIEN_KHAI.md)
> - [Cơ sở lý thuyết thuật toán Random Forest](THUAT_TOAN_RANDOM_FOREST.md)
> - [Danh mục tài liệu tham khảo & Dẫn chứng khoa học](../04_tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md)

---

> **Trạng thái**: thiết kế trước khi thực hiện. Các bảng/ví dụ luật là **minh hoạ cấu trúc**; mọi con số độ tin cậy sẽ do mô hình tính (**[CẦN ĐO]**). Mã nguồn ở mục 5 đã được kiểm thử trên dữ liệu giả lập, chưa chạy trên Covertype.

## 1. BẢN CHẤT VÀ Ý NGHĨA CỦA TRÍCH XUẤT LUẬT (RULE EXTRACTION)

### 1.1. Thách thức "Hộp đen" (Black-Box Problem) trong AI Lâm nghiệp
Trong bài toán phân loại kiểu thảm rừng (Forest Cover Type), mô hình tập hợp **Random Forest** `[7]` kỳ vọng cho độ chính xác cao hơn nhiều so với cây đơn lẻ (mục tiêu ≥ 88%, sẽ đo ở Bước 5 của kế hoạch). Tuy nhiên, Random Forest là một mô hình "Hộp đen" (Black-box) gồm hàng trăm cây quyết định độc lập, mỗi cây bỏ một phiếu khác nhau. Kết quả là:
- Người sử dụng không thể nhìn thấy trực tiếp mối quan hệ logic nguyên nhân - kết quả giữa các yếu tố địa hình và loài cây.
- Các nhà quản lý rừng, kiểm lâm viên không thể mang một mô hình máy tính phức tạp ra thực địa không có sóng mạng.
- Ngược lại, một Cây quyết định đơn lẻ (Single Decision Tree) `[12]` tuy có tính minh bạch ("Hộp kính" - White-box) nhưng nếu để phát triển tự do sẽ mọc sâu 25–30 tầng, tạo ra hàng ngàn nhánh vụn vặt bị quá khớp (overfitting), con người không thể tiếp thu được.

### 1.2. Mục tiêu phương pháp
Phương pháp trích xuất luật nhằm mục đích:
1. **Minh bạch hóa tri thức**: Biến đổi các ranh giới phân lớp toán học trừu tượng thành các **mệnh đề logic hình thức $NẾU \dots THÌ \dots$ (If-Then rules)** ngắn gọn, mạch lạc.
2. **Khái quát hóa quy luật sinh thái**: Trích xuất các điều kiện địa hình cốt lõi chi phối sự sống còn của 7 loài cây, đối chiếu trực tiếp với các quy luật phân tầng thực vật học của Peet (1981) `[4]` và Marr (1961) `[5]`.
3. **Ứng dụng thực địa tức thì**: Cung cấp một bảng "Quy tắc vàng If-Then" gọn gàng để bất kỳ ai có số liệu đo đạc GPS và thổ nhưỡng đều có thể suy đoán gần đúng kiểu rừng mà không cần cài đặt phần mềm phức tạp.

```mermaid
flowchart LR
    A["Mô hình & Dữ liệu thực nghiệm"] --> B["Huấn luyện cây đại diện trên dự đoán của RF (độ sâu hạn chế)"]
    B --> C["Thuật toán duyệt đệ quy DFS (Tree Path Traversal)"]
    C --> D["Định lượng: Confidence, Support, Class coverage, Lift, Fidelity"]
    D --> E["Lọc & Tối ưu hóa: Top 'Quy tắc vàng'"]
    E --> F["Ánh xạ ngữ nghĩa sinh thái học thực tế"]
```

---

## 2. NỀN TẢNG TOÁN HỌC VÀ LOGIC MỆNH ĐỀ CỦA CÂY QUYẾT ĐỊNH

Về mặt toán học rời rạc, một Cây quyết định $T = (V, E)$ là một đồ thị có hướng, không chu trình dạng cây nhị phân (Binary Tree):
- Tập đỉnh $V$ gồm hai loại:
  - Các nút nội bộ (Internal nodes) $V_{int}$: Mỗi nút $v \in V_{int}$ gắn với một hàm kiểm tra điều kiện dạng $f(x) \le \theta$ hoặc $f(x) > \theta$.
  - Các nút lá (Leaf/Terminal nodes) $V_{leaf}$: Mỗi nút lá $u \in V_{leaf}$ gắn với một nhãn phân loại $C \in \{1, 2, \dots, K\}$ và một phân phối xác suất xác nghiệm trên các mẫu rơi vào lá đó.
- Tập cạnh $E$: Kết nối từ nút cha đến hai nút con tương ứng với kết quả Đúng (True) hoặc Sai (False) của phép kiểm tra.

Mỗi đường đi (Path) $\mathcal{P}$ duy nhất xuất phát từ nút gốc $v_{root}$ đi qua các nút nội bộ $v_1, v_2, \dots, v_m$ và kết thúc tại nút lá $u_k$ biểu diễn một **hội logic (Conjunction of Literals)**:

$$\mathcal{P}: \quad \bigwedge_{j=1}^{m} \left( X_{a_j} \text{ op}_j \theta_j \right) \implies \text{Class} = C_k$$

Trong đó:
- $X_{a_j}$ là đặc trưng địa hình được kiểm tra tại bước thứ $j$ (như `Elevation`, `Slope`, `Soil_Type`).
- $\text{op}_j \in \{\le, >\}$ là toán tử quan hệ.
- $\theta_j$ là ngưỡng số thực tối ưu tìm được qua phép tối đa hóa mức giảm độ vẩn đục Gini `[12]`.
- $C_k$ là kiểu thảm rừng chiếm đa số tại nút lá $u_k$.

---

## 3. QUY TRÌNH 5 BƯỚC TRÍCH XUẤT LUẬT CHI TIẾT

### Bước 1: Huấn luyện cây đại diện (Surrogate Tree) có kiểm soát độ phức tạp
1. **Giới hạn độ sâu (`max_depth = 4` đến `6`)**: mỗi đường đi chứa tối đa 4–6 điều kiện để con người đọc được (lựa chọn thực dụng, không phải ngưỡng khoa học cố định).
2. **Số mẫu tối thiểu tại lá (`min_samples_leaf = 100` đến `500`)**: loại các lá cá biệt. Cảnh báo: với lớp hiếm, giá trị lớn có thể làm biến mất hoàn toàn nhánh của Lớp 4. Vì vậy dùng bể truy vấn lớn (phần 85% chưa dùng huấn luyện, có ≈ 2.335 mẫu lớp 4), kiểm tra rằng **cả 7 lớp đều có ít nhất một luật**, và giảm `min_samples_leaf` nếu cần.
3. **Trọng số lớp (`class_weight='balanced'`, tuỳ chọn so sánh với `None`)**: giúp cây không bỏ qua Lớp 4. Vì trọng số làm thay đổi `tree_.value`, mọi chỉ số chất lượng ở Bước 3 được **đếm trên mẫu thực**, không đọc từ `tree_.value`.
4. **Nhãn huấn luyện là dự đoán của Random Forest, không phải nhãn thật**: cây đại diện phải xấp xỉ *hàm quyết định của RF* `[11]`. RF dự đoán lại chính tập huấn luyện của nó thường gần trùng nhãn thật, nên dữ liệu truy vấn RF cần lấy từ phần dữ liệu **chưa dùng để huấn luyện RF**. (Chế độ `dung_nhan_rf=False` — cây học nhãn thật — được giữ để so sánh.)
5. **Báo cáo Fidelity** = tỷ lệ cây đại diện dự đoán trùng RF trên tập độc lập, cùng độ chính xác của cây so với nhãn thật. Fidelity thấp nghĩa là luật chỉ mô tả một phần hành vi của RF và phải nêu rõ điều đó trong báo cáo. Ngưỡng chấp nhận do nhóm quyết định (đề xuất tham khảo ≥ 80%, **[KỲ VỌNG]**).

---

### Bước 2: Thuật toán duyệt đệ quy theo chiều sâu (Depth-First Search - DFS)
Sau khi cây quyết định hội tụ, thuật toán sẽ tự động truy xuất cấu trúc đồ thị bên dưới (trong Scikit-learn là đối tượng `tree_`) để xuất toàn bộ các nhánh rẽ.

**Mã giả thuật toán duyệt cây:**
```text
Thuật_toán Duyệt_Đường_Đi(node_id, danh_sách_điều_kiện_hiện_tại):
    NẾU node_id là Nút_Lá THÌ:
        Tính_toán các chỉ số (Lớp dự đoán, Độ tin cậy, Số mẫu hỗ trợ)
        Lưu luật: [danh_sách_điều_kiện_hiện_tại => Lớp dự đoán]
        TRẢ VỀ
        
    Lấy đặc trưng feature_id và ngưỡng threshold tại node_id
    
    // Rẽ nhánh trái (<= ngưỡng)
    điều_kiện_trái = "feature <= threshold"
    Duyệt_Đường_Đi(child_left, danh_sách_điều_kiện_hiện_tại + [điều_kiện_trái])
    
    // Rẽ nhánh phải (> ngưỡng)
    điều_kiện_phải = "feature > threshold"
    Duyệt_Đường_Đi(child_right, danh_sách_điều_kiện_hiện_tại + [điều_kiện_phải])
```

---

### Bước 3: Định lượng chất lượng của từng luật (Rule Quality Metrics)
Mỗi luật $R$ (một lá của cây) được chấm bằng **đếm mẫu thực** (không trọng số) trên tập độc lập, và trên bể truy vấn để đối chiếu:

1. **Độ tin cậy (Confidence / Precision)**:
   $$Confidence(R) = \frac{N_{\text{đúng}}}{N_{\text{lá}}} = \frac{\text{Số mẫu có nhãn thật là lớp } C \text{ tại lá}}{\text{Tổng số mẫu rơi vào lá}}$$
   *Ví dụ*: $Confidence = 92.5\%$ nghĩa là trong 100 điểm thỏa luật có 92.5 điểm đúng là kiểu rừng đó.
2. **Độ hỗ trợ (Support)**: $Support(R) = \dfrac{N_{\text{lá}}}{N}$. Luật có Support quá nhỏ thường chỉ là hiện tượng cục bộ. **Lưu ý**: Support của một luật lớp 4 không thể vượt $0.473\%$ vì đó là tỷ lệ của cả lớp, nên không dùng Support toàn cục làm tiêu chí loại luật lớp hiếm.
3. **Độ bao phủ lớp (Class coverage / Recall của luật)**: $Coverage(R) = \dfrac{N_{\text{đúng}}}{n_C}$, tỷ lệ mẫu thật của lớp $C$ mà luật bắt được. Đây là thước đo phù hợp cho lớp hiếm.
4. **Độ nâng (Lift)**: $Lift(R) = \dfrac{Confidence(R)}{P(C)}$, với $P(C) = n_C / N$ trên cùng tập. *(Ví dụ số học minh hoạ: lớp 4 với $P(C_4) = 0.0047$ và $Confidence = 70.5\%$ cho $Lift \approx 150$.)*
5. **Độ trung thực của luật (Rule fidelity)**: tỷ lệ mẫu rơi vào lá mà RF dự đoán trùng lớp của luật.

> **Lỗi thường gặp**: `tree_.value` lưu phân phối lớp **đã nhân `class_weight`**, và ở phiên bản Scikit-learn mới đã chuẩn hoá về tổng bằng 1. Nếu dùng nó để tính Confidence/Support thì Support thành $1/N$ cho mọi lá và Confidence bị lệch theo trọng số. Trong kiểm thử trên dữ liệu giả lập mất cân bằng, Confidence suy từ `tree_.value` lệch so với đếm thật tới khoảng $+0.7$ / $-0.2$. Mã ở mục 5 dùng `cay.apply(X)` rồi đếm trực tiếp.

---

### Bước 4: Lọc và chắt lọc thành "Top Quy tắc vàng" (Golden Rules)
Cây độ sâu 5 sinh tối đa 32 luật lá. Để tránh quá tải thông tin:
1. **Lọc ngưỡng chất lượng**: loại luật có $Confidence < 70\%$ (trên tập độc lập) hoặc quá ít mẫu (ví dụ $< 30$ mẫu; ngưỡng đề xuất, có thể điều chỉnh).
2. **Gom nhóm** các luật đạt chuẩn theo 7 kiểu thảm rừng.
3. **Chọn tinh hoa**: với mỗi lớp lấy Top 1–2 luật theo $Confidence \times Coverage$ (thay cho $Confidence \times Support$, vì Support bất lợi cho lớp hiếm), được khoảng 7–10 "Quy tắc vàng".
4. **Lớp không có luật đạt chuẩn**: báo cáo luật tốt nhất kèm cảnh báo, không bỏ lớp.
5. **Luôn kèm Fidelity toàn cục** của cây để người đọc biết luật bao phủ được bao nhiêu hành vi của RF.

---

### Bước 5: Ánh xạ ngữ nghĩa sinh thái học thực tế (Ecological Semantic Mapping)
Các luật trích xuất ban đầu chỉ là các ngưỡng số (ví dụ: `Elevation > 3054.5`). Bước tiếp theo là đối chiếu và "dịch nghĩa" sang các quy luật sinh thái trong Peet (1981) `[4]` và Marr (1961) `[5]`. Bảng dưới **chỉ minh hoạ cách dịch**; ngưỡng thật lấy từ luật do mô hình tìm ra, và ngưỡng văn liệu xem bảng chuẩn ở `BOI_CANH_YEU_CAU_MUC_TIEU.md` (đã xác minh khớp các đai sinh thái chuẩn):

```
[Điều kiện trong luật]       -------->  [Ngữ nghĩa sinh thái (diễn giải)]
Elevation cao (≳ 2.800m)     -------->  Đai rừng cận tuyết (Subalpine)
Elevation thấp (≲ 2.450m)    -------->  Đai chân núi thấp
D_hydro nhỏ                  -------->  Vành đai ẩm ven nguồn nước (Riparian)
Aspect_cos < -0.5            -------->  Sườn hướng Nam, nhiều nắng
Soil_Climatic_Zone = 7 / 8   -------->  Đai thổ nhưỡng Subalpine / Alpine
```
Luật mô tả **tương quan trong dữ liệu Covertype**, không khẳng định quan hệ nhân quả sinh thái.

---

## 4. BẢNG CẤU TRÚC KẾT QUẢ: "BỘ QUY TẮC VÀNG SINH THÁI" (MẪU TRÌNH BÀY)

Bảng dưới là **khung trình bày**. Ngưỡng độ cao là dải tham khảo từ bảng chuẩn (có chồng lấn); cột độ tin cậy sẽ được điền từ kết quả chạy thật.

| Kiểu thảm rừng | Mệnh đề logic (If-Then) — minh hoạ | Confidence / Coverage / Fidelity | Ý nghĩa sinh thái (diễn giải) `[4, 5]` |
| :--- | :--- | :---: | :--- |
| **1. Spruce/Fir** | **NẾU** Độ cao ≈ 2.800–3.400m **VÀ** hướng sườn mát **VÀ** đai thổ nhưỡng Subalpine | [CẦN ĐO] | Đai cận tuyết, ưa lạnh; chồng lấn mạnh với Lodgepole nên cần thêm điều kiện phân biệt. |
| **2. Lodgepole Pine** | **NẾU** Độ cao ≈ 2.600–3.100m **VÀ** đất thoát nước trung bình | [CẦN ĐO] | Loài tiên phong sau cháy, dải độ cao trung gian. |
| **3. Ponderosa Pine** | **NẾU** Độ cao ≈ 1.800–2.600m **VÀ** sườn Nam (`Aspect_cos` thấp) | [CẦN ĐO] | Chịu hạn, sườn thấp khô nóng. |
| **4. Cottonwood/Willow** | **NẾU** Độ cao < 2.450m **VÀ** $D_{\text{hydro}}$ nhỏ | [CẦN ĐO] | Ưa ẩm, thung lũng ven nguồn nước. |
| **5. Aspen** | **NẾU** Độ cao ≈ 2.400–2.900m **VÀ** điều kiện đất (xác định từ dữ liệu) | [CẦN ĐO] | Ưa ẩm vừa phải, đất giàu dinh dưỡng. |
| **6. Douglas-fir** | **NẾU** Độ cao ≈ 2.300–2.800m **VÀ** sườn khuất nắng | [CẦN ĐO] | Cần độ ẩm không khí cao hơn Ponderosa. |
| **7. Krummholz** | **NẾU** Độ cao > 3.250m **VÀ** đai thổ nhưỡng Alpine | [CẦN ĐO] | Ranh giới cây gỗ (tree line). |

*(Các điều kiện chỉ để minh hoạ cấu trúc; luật thật do mô hình tính tại Bước 6.)*

---

## 5. MÃ NGUỒN TRIỂN KHAI PYTHON (SCIKIT-LEARN)

Hàm dưới đây đã được kiểm thử trên dữ liệu giả lập 7 lớp mất cân bằng: tổng `n_la` bằng số mẫu, tổng Support bằng 1, và Confidence khớp với việc tự áp dụng điều kiện luật lên dữ liệu (sai lệch nhỏ chỉ do ngưỡng hiển thị làm tròn 2 chữ số). Chưa chạy trên Covertype.

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def trich_xuat_luat(rf_model, X_fit, y_fit, X_eval, y_eval, feature_names, class_names,
                    max_depth=5, min_samples_leaf=150, class_weight="balanced",
                    dung_nhan_rf=True, random_state=42):
    """Huấn luyện cây đại diện (surrogate) và trích xuất luật If-Then.

    X_fit, y_fit : dữ liệu để HUẤN LUYỆN CÂY ĐẠI DIỆN. Khuyến nghị dùng phần dữ liệu CHƯA dùng
                   để huấn luyện RF (RF dự đoán lại chính tập huấn luyện của nó thường gần
                   trùng nhãn thật, nên cây sẽ không học được hành vi riêng của RF).
    X_eval, y_eval: tập độc lập để chấm điểm luật (test hoặc hold-out lớn).
    class_names  : dict {nhãn: tên lớp}, ví dụ {1: "Spruce/Fir", 2: "Lodgepole Pine", ...}.
    dung_nhan_rf : True  -> cây học theo DỰ ĐOÁN của RF (surrogate đúng nghĩa);
                   False -> cây học theo nhãn thật (cây nông thông thường, để so sánh).

    Mọi chỉ số được ĐẾM trên mẫu thực (không trọng số). Không dùng tree_.value để tính
    confidence/support vì giá trị này đã nhân class_weight.
    Trả về: (cây, DataFrame luật, dict tổng quan gồm fidelity).
    """
    X_fit, X_eval = np.asarray(X_fit), np.asarray(X_eval)
    y_fit, y_eval = np.asarray(y_fit), np.asarray(y_eval)

    rf_fit = rf_model.predict(X_fit)
    rf_eval = rf_model.predict(X_eval)
    muc_tieu = rf_fit if dung_nhan_rf else y_fit

    cay = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 class_weight=class_weight, random_state=random_state)
    cay.fit(X_fit, muc_tieu)
    t = cay.tree_

    # DFS: duyệt từ gốc đến lá, gom các điều kiện trên đường đi
    dieu_kien = {}

    def dfs(node, conds):
        if t.children_left[node] == -1:
            dieu_kien[node] = conds
            return
        ten, nguong = feature_names[t.feature[node]], t.threshold[node]
        dfs(t.children_left[node], conds + [f"{ten} <= {nguong:.2f}"])
        dfs(t.children_right[node], conds + [f"{ten} > {nguong:.2f}"])

    dfs(0, [])

    def lop_cua_la(leaf):
        return cay.classes_[np.argmax(t.value[leaf][0])]

    def chi_so(X, y_that, y_rf, hau_to):
        la = cay.apply(X)
        n = len(y_that)
        out = {}
        for leaf in dieu_kien:
            lop = lop_cua_la(leaf)
            m = la == leaf
            n_la = int(m.sum())
            n_lop = int((y_that == lop).sum())
            n_dung = int((y_that[m] == lop).sum())
            conf = n_dung / n_la if n_la else np.nan
            prior = n_lop / n
            out[leaf] = {
                f"n_la_{hau_to}": n_la,
                f"support_{hau_to}": n_la / n,
                f"confidence_{hau_to}": conf,
                f"class_coverage_{hau_to}": n_dung / n_lop if n_lop else np.nan,
                f"lift_{hau_to}": conf / prior if (n_la and prior > 0) else np.nan,
                f"rule_fidelity_{hau_to}": float((y_rf[m] == lop).mean()) if n_la else np.nan,
            }
        return out

    cs_fit = chi_so(X_fit, y_fit, rf_fit, "fit")
    cs_eval = chi_so(X_eval, y_eval, rf_eval, "eval")

    hang = []
    for leaf, conds in dieu_kien.items():
        lop = lop_cua_la(leaf)
        hang.append({"leaf_id": leaf, "luat": " VÀ ".join(conds), "lop_du_doan": lop,
                     "ten_lop": class_names.get(lop, str(lop)), **cs_fit[leaf], **cs_eval[leaf]})
    df = (pd.DataFrame(hang)
            .sort_values(["lop_du_doan", "confidence_eval"], ascending=[True, False])
            .reset_index(drop=True))

    pred_cay = cay.predict(X_eval)
    tong_quan = {
        "fidelity_eval": float((pred_cay == rf_eval).mean()),
        "acc_cay_vs_nhan_that": float((pred_cay == y_eval).mean()),
        "acc_rf_vs_nhan_that": float((rf_eval == y_eval).mean()),
        "so_luat": len(df),
    }
    return cay, df, tong_quan
```

**Cách dùng** (tên biến minh hoạ):

```python
CLASS_NAMES = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
               5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
# X_pool, y_pool: tối đa vài chục nghìn mẫu lấy từ phần 85% chưa dùng huấn luyện RF (đã qua cùng pipeline đặc trưng)
# X_test, y_test: tập kiểm thử độc lập
cay, df_luat, tong_quan = trich_xuat_luat(rf, X_pool, y_pool, X_test, y_test, FEATURE_NAMES, CLASS_NAMES)
print(tong_quan)   # fidelity_eval, acc_cay_vs_nhan_that, acc_rf_vs_nhan_that, so_luat
```

Lưu ý: ngưỡng trong cột `luat` được làm tròn 2 chữ số chỉ để hiển thị; các chỉ số được tính theo lá của cây, không theo văn bản đã làm tròn.

---

## 6. GIÁ TRỊ KỲ VỌNG VÀ HẠN CHẾ

1. **Giá trị kỳ vọng**: đi từ mô hình hộp đen đến tập luật tường minh, có đo Confidence/Coverage/Lift và **Fidelity** so với RF, thay vì chỉ gọi hàm thư viện.
2. **Hạn chế cần nêu rõ**: (a) cây đại diện nông chỉ *xấp xỉ* RF, nên luật chỉ mô tả một phần hành vi của RF (Fidelity); (b) luật mô tả tương quan, không phải quan hệ nhân quả; (c) dữ liệu Covertype chỉ gồm một vùng ở Colorado, nên luật không mặc nhiên đúng cho vùng khác.
3. **Đối chiếu yêu cầu gốc**: đề bài gốc nằm ở [`docs/detai.md`](../detai.md) (yêu cầu: phân loại 7 kiểu thảm rừng, xử lý dữ liệu lớn bằng lấy mẫu, luật sinh thái rừng, thực hiện trên Google Colab).
4. **[SỐ ĐO] E6**: mô hình chủ có Macro-F1 ≈ 0,28 trên vùng hoang dã chưa thấy (so với 0,97 khi chia ngẫu nhiên; baseline E0 đạt 0,32). Luật xấp xỉ mô hình này vì vậy mô tả hành vi trong phạm vi 4 vùng nghiên cứu, củng cố hạn chế (c).

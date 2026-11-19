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

import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from trich_xuat_luat import trich_xuat_luat

X, y = make_classification(n_samples=40000, n_features=8, n_informative=6, n_redundant=0, n_classes=7,
                           n_clusters_per_class=1, weights=[.35,.40,.06,.01,.03,.08,.07], class_sep=1.5, random_state=0)
y = y + 1
names = [f"f{i}" for i in range(8)]
X_rf, y_rf_ = X[:6000], y[:6000]          # dùng huấn luyện RF (~ "mẫu 15%")
X_fit, y_fit = X[6000:26000], y[6000:26000]  # "bể truy vấn" chưa dùng huấn luyện RF
X_ev, y_ev = X[26000:], y[26000:]            # đánh giá
rf = RandomForestClassifier(n_estimators=60, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1).fit(X_rf, y_rf_)
cn = {i: f"lop{i}" for i in range(1, 8)}
cay, df, tq = trich_xuat_luat(rf, X_fit, y_fit, X_ev, y_ev, names, cn, max_depth=5, min_samples_leaf=150)
print(tq); print("số lá:", len(df), "| số lớp được dự đoán:", sorted(df.lop_du_doan.unique()))

# 1) kiểm tra tính nhất quán
assert df.n_la_eval.sum() == len(y_ev) and abs(df.support_eval.sum() - 1) < 1e-9
assert df.n_la_fit.sum() == len(y_fit)
# 2) kiểm tra độc lập một luật: tự áp dụng điều kiện dạng văn bản (đã làm tròn 2 chữ số)
Xd = pd.DataFrame(X_ev, columns=names)
sai_lech = []
for _, r in df.iterrows():
    mask = np.ones(len(Xd), bool)
    for c in r.luat.split(" VÀ "):
        ten, op, ng = c.split(); ng = float(ng)
        mask &= (Xd[ten] <= ng) if op == "<=" else (Xd[ten] > ng)
    n_vb = int(mask.sum()); n_cay = int(r.n_la_eval)
    conf_vb = (y_ev[mask] == r.lop_du_doan).mean() if n_vb else np.nan
    sai_lech.append(abs(n_vb - n_cay))
    assert abs(conf_vb - r.confidence_eval) < 0.02 or n_cay < 30, (r.luat, conf_vb, r.confidence_eval)
print("lệch số mẫu tối đa do làm tròn ngưỡng 2 chữ số:", max(sai_lech))
# 3) minh hoạ lỗi cũ: confidence tính từ tree_.value (đã nhân trọng số) khác confidence thật
t = cay.tree_; diff = []
for _, r in df.iterrows():
    v = t.value[int(r.leaf_id)][0]; diff.append(v.max() / v.sum() - r.confidence_fit)
print("chênh lệch conf 'tree_.value' so với conf đếm thật (min/max):", round(min(diff), 3), round(max(diff), 3),
      "| tổng tree_.value một lá =", round(t.value[int(df.leaf_id[0])][0].sum(), 3))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(df[["ten_lop", "luat", "n_la_eval", "support_eval", "confidence_eval", "class_coverage_eval", "lift_eval", "rule_fidelity_eval"]].head(8).round(3).to_string())
# 4) chế độ so sánh: cây học nhãn thật
_, df2, tq2 = trich_xuat_luat(rf, X_fit, y_fit, X_ev, y_ev, names, cn, dung_nhan_rf=False)
print("cây học nhãn thật:", tq2)
print("TEST OK")

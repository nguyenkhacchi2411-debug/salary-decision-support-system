"""
pages/6_LR_vs_BLR.py — Trang 6: So sánh LR vs BLR
Đánh giá head-to-head trên cùng TEST SET.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve, auc as sk_auc
from sklearn.calibration import calibration_curve
from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="6 · LR vs BLR", page_icon="⚖️", layout="wide")
st.title("Trang 6 — So sánh LR vs Bayesian LR")
st.caption("Head-to-head trên cùng Test Set · ROC · Calibration · Uncertainty")

feat, src = L.features_or_none()
if feat is None:
    st.error(src); st.stop()

with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        C_val = st.select_slider("C (LR)", [0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0], value=1.0, key="p6_C")
        max_iter = st.slider("max_iter (LR)", 100, 1000, 500, 100, key="p6_iter")
    with col2:
        prior_sigma = st.slider("Prior scale σ (BLR)", 0.1, 10.0, 1.0, 0.1, key="p6_sigma")
        M = st.slider("M — MC samples", 100, 5000, 1000, 100, key="p6_M")

fkey = L.feat_cache_key()
with st.spinner("Train LR + BLR..."):
    lr_b = L.get_lr_bundle(fkey, C=C_val, max_iter=max_iter)
    blr_b = L.get_blr_bundle(fkey, prior_sigma=prior_sigma, C=C_val, max_iter=max_iter)
    pp = P.posterior_predictive(blr_b["posterior"]["mean"], blr_b["posterior"]["cov"],
                                blr_b["X_test_aug"], M=M)

sp = lr_b["split"]
y_test = sp["y_test"]
lr_prob = lr_b["test_prob"]
blr_prob = pp["pred_mean"]

# Threshold tối ưu cho mỗi model (trên val)
lr_opt = P.optimize_threshold_f2(sp["y_val"], lr_b["val_prob"])
blr_pp_val = P.posterior_predictive(blr_b["posterior"]["mean"], blr_b["posterior"]["cov"],
                                    blr_b["X_val_aug"], M=M)
blr_opt = P.optimize_threshold_f2(sp["y_val"], blr_pp_val["pred_mean"])

lr_tau  = lr_opt["best_threshold"]
blr_tau = blr_opt["best_threshold"]
lr_m  = P.classification_metrics(y_test, lr_prob,  lr_tau)
blr_m = P.classification_metrics(y_test, blr_prob, blr_tau)

# ── 1. Bảng so sánh tổng quát ─────────────────────────────────────────────────
st.subheader("1. Overall Performance (cùng Test Set)")
st.caption(f"LR: τ={lr_tau:.2f} | BLR: τ={blr_tau:.2f}  (đều tối ưu F2 trên val)")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Bayesian LR"],
    "Threshold τ": [lr_tau, blr_tau],
    "Accuracy":    [round(lr_m["accuracy"],4),  round(blr_m["accuracy"],4)],
    "Precision":   [round(lr_m["precision"],4), round(blr_m["precision"],4)],
    "Recall":      [round(lr_m["recall"],4),    round(blr_m["recall"],4)],
    "F2 ⭐":       [round(lr_m["f2"],4),         round(blr_m["f2"],4)],
    "ROC-AUC":     [round(lr_m["roc_auc"],4),   round(blr_m["roc_auc"],4)],
    "Brier Score": [round(lr_m["brier"],4),     round(blr_m["brier"],4)],
    "Có Uncertainty?": ["Không ✗", "Có ✓"],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Delta
st.markdown("**Δ = BLR − LR:**")
delta_cols = ["Accuracy","Precision","Recall","F2 ⭐","ROC-AUC","Brier Score"]
delta_vals = {c: round(blr_m[c.lower().replace(" ⭐","").replace("-","_").replace(" ","_")],4)
              - round(lr_m[c.lower().replace(" ⭐","").replace("-","_").replace(" ","_")],4)
              for c in ["Accuracy","Precision","Recall"]}
delta_vals["F2"] = round(blr_m["f2"]-lr_m["f2"],4)
delta_vals["ROC-AUC"] = round(blr_m["roc_auc"]-lr_m["roc_auc"],4)
delta_vals["Brier Score"] = round(blr_m["brier"]-lr_m["brier"],4)

dcols = st.columns(6)
notes = {"Brier Score": "↓ = tốt hơn"}
for col, (k,v) in zip(dcols, delta_vals.items()):
    better = v > 0 if k != "Brier Score" else v < 0
    sign = "✓" if better else "✗"
    col.metric(k, f"{v:+.4f}", help=notes.get(k,"↑ = tốt hơn"))

st.info(
    "BLR thường thắng về **Brier Score và Calibration** (xác suất có nghĩa thực tế), "
    "không nhất thiết thắng về F2/Recall — đây là thông điệp đúng của đồ án: "
    "mục tiêu là uncertainty, không phải maximize accuracy."
)

# ── 2. ROC + PR chồng nhau ────────────────────────────────────────────────────
st.subheader("2. ROC & Precision-Recall Curves")
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_prob)
fpr_blr, tpr_blr, _ = roc_curve(y_test, blr_prob)
prec_lr, rec_lr, _   = precision_recall_curve(y_test, lr_prob)
prec_blr,rec_blr, _  = precision_recall_curve(y_test, blr_prob)

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
# ROC
axes1[0].plot(fpr_lr,  tpr_lr,  color="steelblue", lw=2, label=f"LR  (AUC={lr_m['roc_auc']:.3f})")
axes1[0].plot(fpr_blr, tpr_blr, color="tomato",    lw=2, label=f"BLR (AUC={blr_m['roc_auc']:.3f})")
axes1[0].plot([0,1],[0,1],"k--",alpha=0.4,label="Random")
axes1[0].set_xlabel("FPR"); axes1[0].set_ylabel("TPR")
axes1[0].set_title("ROC Curve (LR vs BLR)"); axes1[0].legend(); axes1[0].grid(alpha=0.3)
# PR
axes1[1].plot(rec_lr,  prec_lr,  color="steelblue", lw=2, label=f"LR  (AUPRC={sk_auc(rec_lr,prec_lr):.3f})")
axes1[1].plot(rec_blr, prec_blr, color="tomato",    lw=2, label=f"BLR (AUPRC={sk_auc(rec_blr,prec_blr):.3f})")
axes1[1].axhline(y_test.mean(), color="gray", lw=1, ls="--", label=f"Baseline={y_test.mean():.3f}")
axes1[1].set_xlabel("Recall"); axes1[1].set_ylabel("Precision")
axes1[1].set_title("Precision-Recall Curve"); axes1[1].legend(); axes1[1].grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig1); plt.close(fig1)

# ── 3. Calibration ────────────────────────────────────────────────────────────
st.subheader("3. Calibration — Reliability Diagram")
st.markdown(
    r"Brier Score $= \frac{1}{N}\sum_i(\pi(x_i)-y_i)^2$.  "
    "Reliability Diagram: đường gần y=x → **well-calibrated** (xác suất dự đoán có nghĩa thực tế)."
)
try:
    frac_lr,  mean_lr  = calibration_curve(y_test, lr_prob,  n_bins=10, strategy="uniform")
    frac_blr, mean_blr = calibration_curve(y_test, blr_prob, n_bins=10, strategy="uniform")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot([0,1],[0,1],"k--",alpha=0.5,label="Perfectly calibrated")
    ax2.plot(mean_lr,  frac_lr,  color="steelblue", lw=2, marker="o", label=f"LR  (Brier={lr_m['brier']:.4f})")
    ax2.plot(mean_blr, frac_blr, color="tomato",    lw=2, marker="s", label=f"BLR (Brier={blr_m['brier']:.4f})")
    ax2.set_xlabel("Mean Predicted Probability"); ax2.set_ylabel("Fraction of positives")
    ax2.set_title("Calibration Curve (Reliability Diagram)"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2); plt.close(fig2)
except Exception as e:
    st.warning(f"Không vẽ được calibration curve: {e}")

# ── 4. Uncertainty Analysis (BLR only) ───────────────────────────────────────
st.subheader("4. Uncertainty Analysis — BLR Only")
st.caption("Đây là phần đặc trưng Bayesian mà LR không có.")
blr_var = pp["pred_var"]
blr_ci  = pp["ci_width"]

c1, c2, c3 = st.columns(3)
c1.metric("Pred Var (mean)", f"{blr_var.mean():.6f}")
c2.metric("CI Width (mean)", f"{blr_ci.mean():.4f}")
c3.metric("Pred Var (max)",  f"{blr_var.max():.6f}")

# Var by label
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
for lbl, color, name in [(0,"steelblue","Normal (0)"), (1,"tomato","Strong Mov. (1)")]:
    mask = y_test == lbl
    axes3[0].hist(blr_var[mask], bins=40, alpha=0.6, color=color, density=True, label=name)
axes3[0].set_xlabel("Predictive Variance"); axes3[0].set_ylabel("Density")
axes3[0].set_title("Predictive Variance theo Actual Label"); axes3[0].legend(); axes3[0].grid(alpha=0.3)

# CI width over time
axes3[1].plot(sp["dates_test"], blr_ci, color="orange", lw=0.8, alpha=0.8)
ci_series = pd.Series(blr_ci, index=sp["dates_test"])
axes3[1].plot(ci_series.rolling(20,center=True).mean(), color="red", lw=2, label="Rolling mean 20-day")
axes3[1].set_xlabel("Date"); axes3[1].set_ylabel("CI Width")
axes3[1].set_title("Uncertainty (CI Width) qua thời gian — Test Set"); axes3[1].legend(); axes3[1].grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3); plt.close(fig3)

# Var correct vs incorrect
y_pred_blr = (blr_prob >= blr_tau).astype(int)
correct_mask   = y_pred_blr == y_test
incorrect_mask = ~correct_mask
var_corr   = blr_var[correct_mask].mean()
var_incorr = blr_var[incorrect_mask].mean()
ratio = var_incorr / max(var_corr, 1e-12)
st.metric("Var(Incorrect)/Var(Correct)", f"{ratio:.3f}×",
          help="Kỳ vọng Bayesian: >1 (model uncertain hơn khi sai)")
if ratio > 1:
    st.success(f"✓ Var(Incorrect)={var_incorr:.6f} > Var(Correct)={var_corr:.6f} — BLR tự nhận biết khi nó dễ sai")
else:
    st.warning("⚠ Cần tăng M hoặc kiểm tra lại BLR.")

"""
pages/3_Baseline_LR.py — Trang 3: Baseline Logistic Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc as sk_auc
)
from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="3 · Baseline LR", page_icon="📉", layout="wide")
st.title("Trang 3 — Baseline Logistic Regression")
st.caption("Mô hình tham chiếu · ROC · Confusion Matrix · Threshold F2")

feat, src = L.features_or_none()
if feat is None:
    st.error(src); st.stop()

with st.expander("⚙️ Settings", expanded=True):
    C_val = st.select_slider(
        "C — Regularization strength (log scale)",
        options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        value=1.0,
        help="C lớn → regularization yếu (overfit). C nhỏ → regularization mạnh (underfit). TLR dùng C=1.0 mặc định của sklearn.",
    )
    max_iter = st.slider("max_iter", 100, 1000, 500, step=100)

fkey = L.feat_cache_key()
with st.spinner("Đang train Logistic Regression..."):
    lr_bundle = L.get_lr_bundle(fkey, C=C_val, max_iter=max_iter)

sp = lr_bundle["split"]
val_prob = lr_bundle["val_prob"]
test_prob = lr_bundle["test_prob"]
y_val, y_test = sp["y_val"], sp["y_test"]

# Threshold tối ưu
opt = P.optimize_threshold_f2(y_val, val_prob)
best_tau = opt["best_threshold"]

# ── 1. Hệ số LR ──────────────────────────────────────────────────────────────
st.subheader("1. Hệ số mô hình (θ)")
coef = lr_bundle["coef"]
coef_df = pd.DataFrame({
    "Feature": P.FEATURE_COLS,
    "Coefficient": coef
}).sort_values("Coefficient")

fig0, ax0 = plt.subplots(figsize=(7, 4))
colors_c = ["crimson" if v < 0 else "steelblue" for v in coef_df["Coefficient"]]
ax0.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_c, alpha=0.85)
ax0.axvline(0, color="black", lw=1)
ax0.set_xlabel("Coefficient value"); ax0.set_title("Feature Importance (Logistic Regression)"); ax0.grid(alpha=0.3, axis="x")
plt.tight_layout()
st.pyplot(fig0); plt.close(fig0)
st.caption(f"Intercept: {lr_bundle['intercept']:+.4f}")

# ── 2. Threshold optimization (val) ──────────────────────────────────────────
st.subheader("2. Threshold Optimization trên Validation Set (F2, β=2)")
st.latex(r"F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}")
st.markdown("F2 ưu tiên **Recall gấp đôi Precision** — bỏ sót biến động mạnh (FN) nguy hiểm hơn cảnh báo nhầm (FP).")

fig1, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(opt["thresholds"], opt["f2"],        color="steelblue",    lw=2,   label="F2 (β=2)")
ax1.plot(opt["thresholds"], opt["f1"],        color="mediumpurple", lw=1.5, ls="-.", label="F1")
ax1.plot(opt["thresholds"], opt["precision"], color="coral",        lw=1.5, ls="--", label="Precision")
ax1.plot(opt["thresholds"], opt["recall"],    color="seagreen",     lw=1.5, ls=":",  label="Recall")
ax1.axvline(best_tau, color="navy", lw=1.8, ls="--",
            label=f"Best τ={best_tau:.2f} (F2={opt['best_f2']:.3f})")
ax1.axvline(0.5, color="gray", lw=1, ls="-.", alpha=0.5, label="τ=0.5 (default)")
ax1.scatter([best_tau], [opt["best_f2"]], color="navy", s=80, zorder=5)
ax1.set_xlabel("Threshold τ"); ax1.set_ylabel("Score"); ax1.set_xlim(0.09, 0.91)
ax1.set_ylim(-0.02, 1.02); ax1.legend(ncol=2, fontsize=9); ax1.grid(alpha=0.3)
ax1.set_title("Threshold Optimization — Validation Set")
plt.tight_layout()
st.pyplot(fig1); plt.close(fig1)

# ── 3. Test set metrics ───────────────────────────────────────────────────────
st.subheader("3. Kết quả trên Test Set")
metrics = P.classification_metrics(y_test, test_prob, best_tau)
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Accuracy",    f"{metrics['accuracy']:.4f}")
c2.metric("Precision",   f"{metrics['precision']:.4f}")
c3.metric("Recall",      f"{metrics['recall']:.4f}")
c4.metric("F2 ⭐",       f"{metrics['f2']:.4f}")
c5.metric("ROC-AUC",     f"{metrics['roc_auc']:.4f}")
c6.metric("Brier Score", f"{metrics['brier']:.4f}", help="Thấp hơn = tốt hơn")
st.caption(f"Threshold: τ={best_tau:.2f} (tối ưu F2 trên val) · Test: {sp['dates_test'][0].date()} → {sp['dates_test'][-1].date()}")

# ── 4. ROC + PR + Confusion Matrix ───────────────────────────────────────────
st.subheader("4. ROC · Precision-Recall · Confusion Matrix")
y_pred_test = (test_prob >= best_tau).astype(int)
cm = confusion_matrix(y_test, y_pred_test)
fpr, tpr, _ = roc_curve(y_test, test_prob)
prec_c, rec_c, _ = precision_recall_curve(y_test, test_prob)

fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

# Confusion matrix
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["Normal","Strong Mov."])
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(["Normal","Strong Mov."])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion Matrix (τ={best_tau:.2f})")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i,j]), ha="center", va="center",
                     fontsize=14, color="white" if cm[i,j] > cm.max()/2 else "black")

# ROC
axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"LR (AUC={metrics['roc_auc']:.3f})")
axes[1].plot([0,1],[0,1],"k--",alpha=0.5,label="Random")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC Curve")
axes[1].legend(); axes[1].grid(alpha=0.3)

# PR
auprc = sk_auc(rec_c, prec_c)
axes[2].plot(rec_c, prec_c, color="coral", lw=2, label=f"LR (AUPRC={auprc:.3f})")
axes[2].axhline(y_test.mean(), color="gray", lw=1, ls="--",
                label=f"Baseline={y_test.mean():.3f}")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title("Precision-Recall Curve")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig2); plt.close(fig2)

st.info(
    "Mô hình này là **baseline tham chiếu** — không có uncertainty. "
    "Trang 4 sẽ thêm lớp Bayesian để định lượng độ tin cậy từng dự đoán."
)

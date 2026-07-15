"""
pages/5_Confidence_Threshold.py — Trang 5: Confidence Score & Threshold
Đánh giá trên TEST SET.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="5 · Confidence & Threshold", page_icon="🎚️", layout="wide")
st.title("Trang 5 — Confidence Score & Threshold")
st.caption("Confidence = 1 − Uncertainty · Accuracy theo bucket · Threshold slider · Test Set")

feat, src = L.features_or_none()
if feat is None:
    st.error(src); st.stop()

with st.expander("⚙️ Settings", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        prior_sigma = st.slider("Prior scale σ", 0.1, 10.0, 1.0, 0.1,
                                key="p5_sigma")
        M = st.slider("M — MC samples", 100, 5000, 1000, 100, key="p5_M")
    with col2:
        threshold = st.slider(
            "Decision threshold τ",
            min_value=0.05, max_value=0.70, value=0.17, step=0.01,
            help="Tối ưu τ=0.17 trên val. Kéo để xem tác động realtime trên test."
        )
        cred_level = st.selectbox("Credible Interval level", [0.80,0.90,0.95,0.99],
                                  index=2, format_func=lambda x: f"{int(x*100)}%",
                                  key="p5_ci")

fkey = L.feat_cache_key()
with st.spinner("Đang tính BLR..."):
    blr = L.get_blr_bundle(fkey, prior_sigma=prior_sigma, C=1.0, max_iter=500)
    pp = P.posterior_predictive(blr["posterior"]["mean"], blr["posterior"]["cov"],
                                blr["X_test_aug"], M=M, cred_level=cred_level)

sp = blr["split"]
y_test = sp["y_test"]
dates_test = sp["dates_test"]
pred_mean = pp["pred_mean"]
pred_var  = pp["pred_var"]
ci_width  = pp["ci_width"]

# Confidence score
confidence, ci_info = P.confidence_from_ciwidth(ci_width)
buckets, bk_info = P.confidence_buckets(confidence)

# ── 1. Công thức Confidence ───────────────────────────────────────────────────
st.subheader("1. Công thức Confidence Score")
st.markdown(r"""
$$U_i = \text{CI\_Width}_i \quad \text{(uncertainty thô)}$$

$$\hat U_i = \text{clip}\!\left(\frac{U_i - U_{2.5\%}}{U_{97.5\%} - U_{2.5\%}},\;0,\;1\right) \in [0,1]$$

$$C_i = 1 - \hat U_i \in [0,1]$$

**Bucket (tertile data-driven):**  High ≥ p67 · Low < p33 · còn lại Medium
""")
st.caption(
    f"Robust quantile normalization: p2.5={ci_info['p2.5']:.4f}, p97.5={ci_info['p97.5']:.4f}  |  "
    f"Bucket thresholds: p33={bk_info['p33']:.4f}, p67={bk_info['p67']:.4f}"
)

# ── 2. Phân phối Confidence ───────────────────────────────────────────────────
st.subheader("2. Phân phối Confidence Score")
import collections
bk_counts = collections.Counter(buckets)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Confidence (mean)", f"{confidence.mean():.4f}")
c2.metric("High bucket", f"{bk_counts.get('High',0)}")
c3.metric("Medium bucket", f"{bk_counts.get('Medium',0)}")
c4.metric("Low bucket", f"{bk_counts.get('Low',0)}")

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
axes1[0].hist(confidence, bins=50, color="steelblue", alpha=0.85)
axes1[0].axvline(bk_info["p33"], color="red", ls="--", label=f"p33={bk_info['p33']:.2f}")
axes1[0].axvline(bk_info["p67"], color="green", ls="--", label=f"p67={bk_info['p67']:.2f}")
axes1[0].set_xlabel("Confidence Score"); axes1[0].set_ylabel("Count")
axes1[0].set_title("Phân phối Confidence"); axes1[0].legend(); axes1[0].grid(alpha=0.3)

bucket_labels = ["High","Medium","Low"]
bucket_vals = [bk_counts.get(b,0) for b in bucket_labels]
axes1[1].bar(bucket_labels, bucket_vals, color=["seagreen","coral","tomato"], alpha=0.85)
for i,v in enumerate(bucket_vals):
    axes1[1].text(i, v+1, str(v), ha="center", fontsize=11)
axes1[1].set_ylabel("Count"); axes1[1].set_title("Số quan sát theo bucket"); axes1[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
st.pyplot(fig1); plt.close(fig1)

# ── 3. Accuracy theo Confidence Bucket ───────────────────────────────────────
st.subheader("3. Accuracy theo Confidence Bucket")
st.markdown(
    "**Giả thuyết cốt lõi:** nếu Confidence Score có ý nghĩa thực sự, ta kỳ vọng:  \n"
    r"$$\text{Acc}(\text{High}) > \text{Acc}(\text{Medium}) > \text{Acc}(\text{Low})$$"
)

y_pred = (pred_mean >= threshold).astype(int)
correct = (y_pred == y_test)

bucket_stats = []
for bk in ["High", "Medium", "Low"]:
    mask = buckets == bk
    n = mask.sum()
    if n == 0:
        continue
    acc = correct[mask].mean()
    mean_var = pred_var[mask].mean()
    mean_conf = confidence[mask].mean()
    n_pos = y_test[mask].sum()
    bucket_stats.append({
        "Bucket": bk, "N": int(n),
        "Accuracy": round(float(acc), 4),
        "Mean Confidence": round(float(mean_conf), 4),
        "Mean Pred Var": round(float(mean_var), 6),
        "Label=1 (%)": round(float(n_pos/n*100), 1),
    })

stats_df = pd.DataFrame(bucket_stats)
st.dataframe(stats_df, use_container_width=True, hide_index=True)

accs = [r["Accuracy"] for r in bucket_stats]
mono_ok = all(accs[i] >= accs[i+1] for i in range(len(accs)-1))
if mono_ok:
    st.success(f"✓ Monotonicity Accuracy: {' > '.join(str(a) for a in accs)} — ĐÚNG")
else:
    st.warning(f"⚠ Monotonicity Accuracy: {' vs '.join(str(a) for a in accs)} — không hoàn toàn (thử tăng M)")

fig2, ax2 = plt.subplots(figsize=(7, 4))
colors_bk = {"High":"seagreen","Medium":"coral","Low":"tomato"}
for row in bucket_stats:
    ax2.bar(row["Bucket"], row["Accuracy"], color=colors_bk.get(row["Bucket"],"gray"), alpha=0.85)
    ax2.text(row["Bucket"], row["Accuracy"]+0.01, f"{row['Accuracy']:.3f}", ha="center", fontsize=11)
ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1.1)
ax2.set_title("Accuracy theo Confidence Bucket"); ax2.grid(alpha=0.3, axis="y")
plt.tight_layout()
st.pyplot(fig2); plt.close(fig2)

# ── 4. Threshold slider → realtime metrics ────────────────────────────────────
st.subheader(f"4. Realtime Metrics (τ={threshold:.2f})")
m = P.classification_metrics(y_test, pred_mean, threshold)
cc1,cc2,cc3,cc4,cc5 = st.columns(5)
cc1.metric("Accuracy",  f"{m['accuracy']:.4f}")
cc2.metric("Precision", f"{m['precision']:.4f}")
cc3.metric("Recall",    f"{m['recall']:.4f}")
cc4.metric("F2 ⭐",     f"{m['f2']:.4f}")
cc5.metric("Brier",     f"{m['brier']:.4f}")

# Threshold sweep chart
thresholds_sweep = np.arange(0.05, 0.71, 0.01)
f2s = [P.classification_metrics(y_test, pred_mean, t)["f2"] for t in thresholds_sweep]
recs = [P.classification_metrics(y_test, pred_mean, t)["recall"] for t in thresholds_sweep]
prcs = [P.classification_metrics(y_test, pred_mean, t)["precision"] for t in thresholds_sweep]

fig3, ax3 = plt.subplots(figsize=(11, 4))
ax3.plot(thresholds_sweep, f2s,  color="steelblue",  lw=2,   label="F2 (β=2)")
ax3.plot(thresholds_sweep, recs, color="seagreen",   lw=1.5, ls=":", label="Recall")
ax3.plot(thresholds_sweep, prcs, color="coral",      lw=1.5, ls="--", label="Precision")
ax3.axvline(threshold, color="navy", lw=1.8, ls="--", label=f"τ={threshold:.2f} (hiện tại)")
ax3.set_xlabel("Threshold τ"); ax3.set_ylabel("Score")
ax3.set_title("BLR — Metrics theo threshold (Test Set)"); ax3.legend(ncol=2); ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3); plt.close(fig3)

# ── 5. Confidence theo thời gian ──────────────────────────────────────────────
st.subheader("5. Confidence Score theo thời gian")
fig4, ax4 = plt.subplots(figsize=(13, 4))
colors_t = {"High":"seagreen","Medium":"coral","Low":"tomato"}
for bk in ["High","Medium","Low"]:
    mask = buckets == bk
    ax4.scatter(dates_test[mask], confidence[mask],
                color=colors_t[bk], s=12, alpha=0.7, label=bk)
ax4.set_ylabel("Confidence"); ax4.set_title("Confidence Score theo thời gian (Test Set)")
ax4.legend(); ax4.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig4); plt.close(fig4)

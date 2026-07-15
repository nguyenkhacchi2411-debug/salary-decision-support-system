"""
pages/7_Uncertainty_Impact.py — Trang 7: Uncertainty Impact Analysis
Kiểm chứng 4 giả thuyết cốt lõi. Đánh giá trên TEST SET.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats as scipy_stats
from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="7 · Uncertainty Impact", page_icon="🔬", layout="wide")
st.title("Trang 7 — Uncertainty Impact Analysis")
st.caption("Uncertainty có ý nghĩa thực sự? · 4 giả thuyết · Test Set")

feat, src = L.features_or_none()
if feat is None:
    st.error(src); st.stop()

with st.expander("⚙️ Settings", expanded=False):
    prior_sigma = st.slider("Prior scale σ", 0.1, 10.0, 1.0, 0.1, key="p7_sigma")
    M = st.slider("M — MC samples", 100, 5000, 1000, 100, key="p7_M")
    threshold = st.slider("Decision threshold τ", 0.05, 0.70, 0.17, 0.01, key="p7_tau")

fkey = L.feat_cache_key()
with st.spinner("Đang tính BLR + Posterior Predictive..."):
    blr_b = L.get_blr_bundle(fkey, prior_sigma=prior_sigma, C=1.0, max_iter=500)
    pp = P.posterior_predictive(blr_b["posterior"]["mean"], blr_b["posterior"]["cov"],
                                blr_b["X_test_aug"], M=M)

sp = blr_b["split"]
y_test = sp["y_test"]
dates_test = sp["dates_test"]
pred_mean = pp["pred_mean"]
pred_var  = pp["pred_var"]
ci_width  = pp["ci_width"]

confidence, ci_info = P.confidence_from_ciwidth(ci_width)
buckets, bk_info = P.confidence_buckets(confidence)
y_pred = (pred_mean >= threshold).astype(int)
correct = (y_pred == y_test).astype(int)

# ── Header: 4 câu hỏi ────────────────────────────────────────────────────────
st.subheader("Câu hỏi trung tâm: Uncertainty có ý nghĩa thực sự không?")
st.markdown(
    """
| Section | Kỳ vọng | Ý nghĩa |
|---|---|---|
| **7.1** Confidence Bucket | Acc(High) > Acc(Med) > Acc(Low) | Model biết khi nào nó đáng tin |
| **7.2** Correct vs Incorrect | Var(Incorrect) > Var(Correct) | Var cao → dự báo sai nhiều hơn |
| **7.3** CI Width | Corr(CI, Var) ≈ 1 | Hai đo lường uncertainty nhất quán |
| **7.4** Market Regime | Var(High Vol) > Var(Low Vol) | Model nhạy cảm với thị trường |
"""
)

# ── 7.1 Confidence Bucket ─────────────────────────────────────────────────────
st.subheader("7.1 — Performance theo Confidence Bucket")
rows71 = []
for bk in ["High","Medium","Low"]:
    mask = buckets == bk
    n = mask.sum()
    if n == 0: continue
    acc = correct[mask].mean()
    rows71.append({
        "Bucket":bk, "N":int(n),
        "Accuracy": round(float(acc),4),
        "Mean Confidence": round(float(confidence[mask].mean()),4),
        "Mean Pred Var": round(float(pred_var[mask].mean()),6),
    })

df71 = pd.DataFrame(rows71)
st.dataframe(df71, use_container_width=True, hide_index=True)

accs = [r["Accuracy"] for r in rows71]
mono = all(accs[i] >= accs[i+1] for i in range(len(accs)-1))
if mono:
    st.success(f"✓ Acc đơn điệu: {' > '.join(str(a) for a in accs)} — ĐÚNG")
else:
    st.warning(f"⚠ Acc: {' vs '.join(str(a) for a in accs)} — thử tăng M")

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
colors_bk = {"High":"seagreen","Medium":"coral","Low":"tomato"}
bk_labels = [r["Bucket"] for r in rows71]
bk_accs   = [r["Accuracy"] for r in rows71]
bk_vars   = [r["Mean Pred Var"] for r in rows71]
bars = axes1[0].bar(bk_labels, bk_accs,
                    color=[colors_bk.get(b,"gray") for b in bk_labels], alpha=0.85)
for bar,val in zip(bars,bk_accs):
    axes1[0].text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                  ha="center", fontsize=11)
axes1[0].set_ylabel("Accuracy"); axes1[0].set_ylim(0,1.1)
axes1[0].set_title("Accuracy theo Confidence Bucket"); axes1[0].grid(alpha=0.3,axis="y")

axes1[1].bar(bk_labels, bk_vars,
             color=[colors_bk.get(b,"gray") for b in bk_labels], alpha=0.85)
axes1[1].set_ylabel("Mean Predictive Variance"); axes1[1].set_title("Pred Var theo Bucket")
axes1[1].grid(alpha=0.3, axis="y")
plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

# ── 7.2 Correct vs Incorrect Variance ────────────────────────────────────────
st.subheader("7.2 — Var(Incorrect) vs Var(Correct)")
st.markdown(r"Kỳ vọng: $\mathbb{E}[\text{Var}(\pi) \mid \hat y \neq y] > \mathbb{E}[\text{Var}(\pi) \mid \hat y = y]$")

var_corr   = pred_var[correct==1]
var_incorr = pred_var[correct==0]
u_stat, p_val = scipy_stats.mannwhitneyu(var_incorr, var_corr, alternative="greater")
ratio_72 = var_incorr.mean() / max(var_corr.mean(), 1e-12)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Var(Correct)",   f"{var_corr.mean():.6f}")
c2.metric("Var(Incorrect)", f"{var_incorr.mean():.6f}")
c3.metric("Ratio Incorrect/Correct", f"{ratio_72:.3f}×")
c4.metric("Mann-Whitney p-value", f"{p_val:.2e}")

if p_val < 0.05 and ratio_72 > 1:
    st.success(f"✓ ĐÚNG — Var(Incorrect) > Var(Correct), p={p_val:.2e} < 0.05")
else:
    st.warning("⚠ Không có ý nghĩa thống kê — thử tăng M hoặc kiểm tra BLR.")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
axes2[0].hist(var_corr,   bins=40, alpha=0.6, color="steelblue", density=True, label=f"Correct (n={len(var_corr)})")
axes2[0].hist(var_incorr, bins=40, alpha=0.6, color="tomato",    density=True, label=f"Incorrect (n={len(var_incorr)})")
axes2[0].set_xlabel("Predictive Variance"); axes2[0].set_ylabel("Density")
axes2[0].set_title("Pred Var: Correct vs Incorrect"); axes2[0].legend(); axes2[0].grid(alpha=0.3)

axes2[1].boxplot([var_corr, var_incorr], labels=["Correct","Incorrect"],
                 patch_artist=True,
                 boxprops=dict(facecolor="steelblue"),
                 medianprops=dict(color="red",lw=2))
axes2[1].set_ylabel("Predictive Variance")
axes2[1].set_title(f"Boxplot Variance (ratio={ratio_72:.2f}×)"); axes2[1].grid(alpha=0.3)
plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ── 7.3 CI Width ─────────────────────────────────────────────────────────────
st.subheader("7.3 — CI Width & Uncertainty Consistency")
corr_ci_var  = float(np.corrcoef(ci_width, pred_var)[0,1])
corr_ci_conf = float(np.corrcoef(ci_width, confidence)[0,1])

c1,c2,c3 = st.columns(3)
c1.metric("Corr(CI Width, Pred Var)", f"{corr_ci_var:.4f}", help="Kỳ vọng ≈ 1")
c2.metric("Corr(CI Width, Confidence)", f"{corr_ci_conf:.4f}", help="Kỳ vọng ≈ −1")
c3.metric("CI Width Mean", f"{ci_width.mean():.4f}")

if abs(corr_ci_var) > 0.85:
    st.success(f"✓ Hai đo lường uncertainty nhất quán: Corr(CI_Width, Var) = {corr_ci_var:.4f}")
else:
    st.warning(f"Corr thấp hơn kỳ vọng ({corr_ci_var:.4f}) — tăng M để ổn định")

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
axes3[0].scatter(ci_width, pred_var, alpha=0.3, s=8, color="steelblue")
axes3[0].set_xlabel("CI Width"); axes3[0].set_ylabel("Predictive Variance")
axes3[0].set_title(f"CI Width vs Pred Var  (r={corr_ci_var:.4f})"); axes3[0].grid(alpha=0.3)

axes3[1].plot(dates_test, ci_width, color="orange", lw=0.8, alpha=0.8, label="CI Width")
ci_roll = pd.Series(ci_width, index=dates_test).rolling(20,center=True).mean()
axes3[1].plot(ci_roll.index, ci_roll.values, color="red", lw=2, label="Rolling mean 20d")
axes3[1].set_xlabel("Date"); axes3[1].set_ylabel("CI Width")
axes3[1].set_title("Uncertainty qua thời gian (Test Set)"); axes3[1].legend(); axes3[1].grid(alpha=0.3)
plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

# ── 7.4 Market Regime ─────────────────────────────────────────────────────────
st.subheader("7.4 — Market Regime: High vs Low Volatility")
st.markdown(
    r"Kỳ vọng: $\mathbb{E}[\text{Var}(\pi) \mid \text{High Vol}] > \mathbb{E}[\text{Var}(\pi) \mid \text{Low Vol}]$  "
    "— mô hình tự động uncertain hơn khi thị trường biến động mạnh."
)

# Align market volatility with test set
feat_test = feat.loc[sp["dates_test"]] if all(d in feat.index for d in sp["dates_test"]) else None
if feat_test is not None:
    market_vol = feat_test["Volatility"].values
    vol_median = float(np.median(market_vol))
    high_vol_mask = market_vol >= vol_median
    low_vol_mask  = ~high_vol_mask

    var_hi = pred_var[high_vol_mask]
    var_lo = pred_var[low_vol_mask]
    ratio_74 = var_hi.mean() / max(var_lo.mean(), 1e-12)
    pearson_r, p74 = scipy_stats.pearsonr(market_vol, pred_var)
    _, p_mw74 = scipy_stats.mannwhitneyu(var_hi, var_lo, alternative="greater")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Var (High Vol)", f"{var_hi.mean():.6f}")
    c2.metric("Var (Low Vol)",  f"{var_lo.mean():.6f}")
    c3.metric("Ratio",          f"{ratio_74:.3f}×")
    c4.metric("Pearson r(Vol, Var)", f"{pearson_r:.4f}")

    if ratio_74 > 1 and p_mw74 < 0.05:
        st.success(f"✓ ĐÚNG — Var(High Vol) > Var(Low Vol), ratio={ratio_74:.2f}×, p={p_mw74:.2e}")
    else:
        st.warning(f"⚠ ratio={ratio_74:.2f}×, p={p_mw74:.2e}")

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
    axes4[0].hist(var_lo, bins=40, alpha=0.6, color="steelblue", density=True,
                  label=f"Low Vol  (n={low_vol_mask.sum()})")
    axes4[0].hist(var_hi, bins=40, alpha=0.6, color="tomato",    density=True,
                  label=f"High Vol (n={high_vol_mask.sum()})")
    axes4[0].set_xlabel("Predictive Variance"); axes4[0].set_ylabel("Density")
    axes4[0].set_title(f"Pred Var theo Market Regime (median Vol={vol_median:.5f})")
    axes4[0].legend(); axes4[0].grid(alpha=0.3)

    axes4[1].scatter(market_vol, pred_var, alpha=0.3, s=8, color="steelblue")
    axes4[1].set_xlabel("Market Volatility (10-day)"); axes4[1].set_ylabel("Predictive Variance")
    axes4[1].set_title(f"Scatter: Volatility vs Pred Var  (r={pearson_r:.4f})")
    axes4[1].grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)
else:
    st.warning("Không align được Volatility với test set dates — kiểm tra lại index.")

# ── Summary ───────────────────────────────────────────────────────────────────
st.subheader("📋 Tổng kết 4 giả thuyết")
summary = [
    ("7.1 Acc đơn điệu theo bucket", "Acc(High)>Acc(Med)>Acc(Low)", mono),
    ("7.2 Var(Incorrect) > Var(Correct)", f"ratio={ratio_72:.2f}×, p={p_val:.2e}",
     ratio_72 > 1 and p_val < 0.05),
    ("7.3 CI Width nhất quán với Pred Var", f"r={corr_ci_var:.4f}", abs(corr_ci_var) > 0.85),
    ("7.4 Var(High Vol) > Var(Low Vol)",
     f"ratio={ratio_74:.2f}×" if feat_test is not None else "N/A",
     ratio_74 > 1 and p_mw74 < 0.05 if feat_test is not None else False),
]
n_confirmed = sum(r[2] for r in summary)
for name, result, ok in summary:
    icon = "✅" if ok else "❌"
    st.markdown(f"{icon} **{name}** — {result}")

if n_confirmed == 4:
    st.success("✅ 4/4 kỳ vọng được xác nhận → Uncertainty có ý nghĩa thực sự, không phải artifact toán học.")
else:
    st.info(f"{n_confirmed}/4 kỳ vọng xác nhận — thử tăng M hoặc điều chỉnh prior σ.")

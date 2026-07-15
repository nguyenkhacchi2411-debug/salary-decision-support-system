"""
pages/4_Bayesian_LR.py — Trang 4 ⭐: Bayesian Logistic Regression
Trang trọng tâm: Prior → Posterior → Posterior Predictive → Uncertainty
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="4 · Bayesian LR", page_icon="🎯", layout="wide")
st.title("Trang 4 ⭐ — Bayesian Logistic Regression")
st.caption("Prior · Posterior (Laplace) · Posterior Predictive (MC) · Uncertainty")

feat, src = L.features_or_none()
if feat is None:
    st.error(src); st.stop()

with st.expander("⚙️ Settings — Bayesian Hyperparameters", expanded=True):
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        prior_sigma = st.slider(
            "Prior scale σ",
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            help="prior_variance = σ². Lớn → prior yếu (vague), dữ liệu chiếm ưu thế. "
                 "Nhỏ → regularization mạnh. BLR dùng σ=1.0."
        )
    with col_s2:
        M = st.slider(
            "M — Số MC samples (posterior predictive)",
            min_value=100, max_value=5000, value=1000, step=100,
            help="Số lần sample θ từ posterior để xấp xỉ posterior predictive. "
                 "Lớn hơn → ổn định hơn nhưng chậm hơn."
        )
    with col_s3:
        cred_level = st.selectbox(
            "Credible Interval level",
            options=[0.80, 0.90, 0.95, 0.99],
            index=2,
            format_func=lambda x: f"{int(x*100)}%",
            help="Mức độ credible interval của posterior predictive."
        )

fkey = L.feat_cache_key()

with st.spinner("Đang chạy Bayesian Online Update + Laplace..."):
    blr = L.get_blr_bundle(fkey, prior_sigma=prior_sigma, C=1.0, max_iter=500)

post = blr["posterior"]
prior_mean = blr["prior_mean"]
prior_cov = blr["prior_cov"]
sp = blr["split"]

# Posterior predictive trên TEST SET (honest evaluation, đồng bộ NB6)
with st.spinner(f"Đang tính Posterior Predictive (M={M})..."):
    pp = P.posterior_predictive(
        post["mean"], post["cov"],
        blr["X_test_aug"], M=M, cred_level=cred_level
    )

# ── 0. Lý thuyết tóm tắt ─────────────────────────────────────────────────────
with st.expander("📖 Lý thuyết Bayesian Logistic Regression", expanded=False):
    st.markdown(r"""
**Mô hình:**
$$P(y=1 \mid x, \theta) = \sigma(\theta^\top x), \quad \sigma(z)=\frac{1}{1+e^{-z}}$$

**Prior:**
$$p(\theta) = \mathcal{N}(\mu_0, \Sigma_0), \quad \mu_0[\text{intercept}] = \log\frac{p(y=1)}{p(y=0)}, \quad \mu_0[\text{features}] = 0$$

**Posterior (Laplace Approximation):**
$$p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta_{\text{MAP}},\; H^{-1})$$
$$H = X^\top W X + \Sigma'^{-1}, \quad W = \text{diag}(p_i(1-p_i))$$

**Bayesian Online Update** (Discount λ=0.95 mỗi batch):
$$\Sigma' = \Sigma/\lambda \quad \text{(nới lỏng prior — forgetting mechanism)}$$

**Posterior Predictive (Monte Carlo M samples):**
$$\bar\pi(x^*) = \frac{1}{M}\sum_{m=1}^M \sigma(x^{*\top}\theta^{(m)}), \quad \theta^{(m)} \sim \mathcal{N}(\theta_{\text{MAP}}, H^{-1})$$

**Credible Interval** (không phải Confidence Interval frequentist):
$$\text{CI}_{(1-\alpha)} = [Q_{\alpha/2}(\pi_m),\; Q_{1-\alpha/2}(\pi_m)]$$
""")

# ── 1. Prior Initialization ───────────────────────────────────────────────────
st.subheader("1. Prior Distribution")
prior_std = np.sqrt(np.diag(prior_cov))
prior_df = pd.DataFrame({
    "Parameter": P.PARAM_LABELS,
    "Prior Mean μ₀": prior_mean.round(4),
    "Prior Std σ₀": prior_std.round(4),
})
st.dataframe(prior_df, use_container_width=True, hide_index=True)
st.caption(
    f"Prior variance = σ² = {prior_sigma:.1f}² = {prior_sigma**2:.2f}  |  "
    f"Intercept = log-odds class balance = {prior_mean[0]:.4f}"
)

# ── 2. Prior vs Posterior ─────────────────────────────────────────────────────
st.subheader("2. Prior vs Posterior — Bayesian Updating Evidence")
post_std = np.sqrt(np.diag(post["cov"]))
post_mean = post["mean"]
reduction = (prior_std - post_std) / prior_std * 100

x_pos = np.arange(len(P.PARAM_LABELS))
fig1, axes1 = plt.subplots(2, 1, figsize=(13, 8))

# Mean shift
bars = axes1[0].bar(x_pos, post_mean, color="steelblue", alpha=0.8, label="Posterior Mean θ_MAP")
axes1[0].axhline(0, color="gray", lw=1.2, ls="--", alpha=0.7, label="Prior Mean μ₀ (=0 cho features)")
axes1[0].errorbar(x_pos, post_mean, yerr=1.96*post_std, fmt="none",
                  color="navy", capsize=4, lw=1.2, label="±1.96 posterior std")
for bar, val in zip(bars, post_mean):
    axes1[0].text(bar.get_x()+bar.get_width()/2, val + np.sign(val)*0.02,
                  f"{val:.3f}", ha="center", fontsize=8)
axes1[0].set_xticks(x_pos); axes1[0].set_xticklabels(P.PARAM_LABELS, rotation=20, ha="right", fontsize=9)
axes1[0].set_title("Posterior Mean vs Prior Mean  (≠0 → feature học được tín hiệu từ dữ liệu)")
axes1[0].set_ylabel("Coefficient"); axes1[0].legend(fontsize=9); axes1[0].grid(alpha=0.3, axis="y")

# Variance reduction
w = 0.38
axes1[1].bar(x_pos - w/2, prior_std, w, color="lightcoral", alpha=0.85, label="Prior std σ₀")
axes1[1].bar(x_pos + w/2, post_std,  w, color="steelblue",  alpha=0.85, label="Posterior std σ_post")
for i, (ps, pos, red) in enumerate(zip(prior_std, post_std, reduction)):
    axes1[1].text(i, max(ps, pos)+0.01, f"-{red:.0f}%", ha="center", fontsize=8, color="darkgreen")
axes1[1].set_xticks(x_pos); axes1[1].set_xticklabels(P.PARAM_LABELS, rotation=20, ha="right", fontsize=9)
axes1[1].set_title("Uncertainty Reduction: Prior σ₀ → Posterior σ  (% = đã học từ dữ liệu)")
axes1[1].set_ylabel("Standard deviation"); axes1[1].legend(fontsize=9); axes1[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
st.pyplot(fig1); plt.close(fig1)

summary_df = pd.DataFrame({
    "Parameter": P.PARAM_LABELS,
    "Prior Mean": prior_mean.round(4),
    "Post Mean": post_mean.round(4),
    "Prior Std": prior_std.round(4),
    "Post Std": post_std.round(4),
    "Reduction %": reduction.round(1),
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── 3. Posterior Evolution ────────────────────────────────────────────────────
st.subheader("3. Posterior Evolution (Bayesian Online Update)")
mean_hist = post["mean_history"]     # (n_batches, n_params)
var_hist  = post["var_history"]      # (n_batches, n_params)
grad_norms = post["grad_norms"]
batches = np.arange(1, post["n_batches"] + 1)

fig2, axes2 = plt.subplots(3, 1, figsize=(13, 10))
colors_p = plt.cm.tab10(np.linspace(0, 1, len(P.PARAM_LABELS)))

for i, (lbl, col) in enumerate(zip(P.PARAM_LABELS, colors_p)):
    lw = 2.2 if i == 0 else 1.3
    axes2[0].plot(batches, mean_hist[:, i], label=lbl, color=col, lw=lw)
axes2[0].axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)
axes2[0].set_title("Posterior Mean θ theo từng batch")
axes2[0].set_ylabel("Posterior Mean"); axes2[0].legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=7)
axes2[0].grid(alpha=0.4)

for i, (lbl, col) in enumerate(zip(P.PARAM_LABELS, colors_p)):
    axes2[1].plot(batches, var_hist[:, i], label=lbl, color=col, lw=1.3, alpha=0.85)
axes2[1].set_title("Posterior Variance σ² theo từng batch  (giảm dần = đang học)")
axes2[1].set_ylabel("Posterior Variance (σ²)"); axes2[1].legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=7)
axes2[1].grid(alpha=0.4)

axes2[2].plot(batches, grad_norms, color="coral", lw=1.5)
axes2[2].axhline(1.0, color="red", ls="--", alpha=0.5, label="||∇||=1")
axes2[2].set_title("MAP Convergence — Gradient Norm cuối mỗi batch")
axes2[2].set_xlabel("Batch"); axes2[2].set_ylabel("||gradient||")
axes2[2].legend(fontsize=9); axes2[2].grid(alpha=0.4)

plt.tight_layout()
st.pyplot(fig2); plt.close(fig2)

var_reduction_pct = (var_hist[0].mean() - var_hist[-1].mean()) / var_hist[0].mean() * 100
st.caption(
    f"Variance: batch 1 mean={var_hist[0].mean():.4f} → batch last mean={var_hist[-1].mean():.4f}  "
    f"| Giảm {var_reduction_pct:.1f}%  |  Gradient cuối: {grad_norms[-1]:.5f} (≈0 = hội tụ)"
)

# ── 4. Posterior Predictive ───────────────────────────────────────────────────
st.subheader(f"4. Posterior Predictive Distribution — Test Set ({int(cred_level*100)}% CI)")
st.markdown(r"""
$$\bar\pi(x^*) = \frac{1}{M}\sum_{m=1}^M \sigma(x^{*\top}\theta^{(m)}), \quad
\text{Var}[\pi] = \frac{1}{M}\sum_m (\pi_m - \bar\pi)^2, \quad
\text{CI} = [Q_{\alpha/2}, Q_{1-\alpha/2}]$$
""")

pred_mean = pp["pred_mean"]
pred_var  = pp["pred_var"]
ci_width  = pp["ci_width"]
dates_test = sp["dates_test"]
y_test = sp["y_test"]

c1, c2, c3 = st.columns(3)
c1.metric("Pred Mean (mean)", f"{pred_mean.mean():.4f}")
c2.metric("Pred Variance (mean)", f"{pred_var.mean():.6f}")
c3.metric(f"CI Width {int(cred_level*100)}% (mean)", f"{ci_width.mean():.4f}")

# Timeseries plot
fig3, axes3 = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
pos_mask = y_test == 1
axes3[0].fill_between(dates_test, pp["ci_lower"], pp["ci_upper"],
                      alpha=0.25, color="tomato", label=f"{int(cred_level*100)}% Credible Interval")
axes3[0].plot(dates_test, pred_mean, color="tomato", lw=1.2, alpha=0.9,
              label="Predictive Mean π̄(x*)")
axes3[0].axhline(0.5, color="navy", lw=1, ls="--", alpha=0.6, label="τ=0.5")
axes3[0].scatter(dates_test[pos_mask], pred_mean[pos_mask],
                 color="green", s=8, alpha=0.5, zorder=3, label="Actual y=1")
axes3[0].set_ylabel("P(y=1 | x, θ)"); axes3[0].set_ylim(-0.02, 1.05)
axes3[0].set_title("Posterior Predictive Mean với Credible Interval (Test Set)")
axes3[0].legend(loc="upper left", fontsize=8, ncol=2); axes3[0].grid(alpha=0.3)

ci_series = pd.Series(ci_width, index=dates_test)
rolling_unc = ci_series.rolling(window=20, center=True).mean()
axes3[1].fill_between(dates_test, 0, ci_width, alpha=0.45, color="orange", label="CI Width = Uncertainty")
axes3[1].plot(rolling_unc.index, rolling_unc.values, color="red", lw=2, label="Rolling mean (20-day)")
axes3[1].set_xlabel("Date"); axes3[1].set_ylabel("CI Width")
axes3[1].set_title("Uncertainty qua thời gian — CI Width cao = mô hình uncertain")
axes3[1].legend(fontsize=9); axes3[1].grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3); plt.close(fig3)

# Predictive distribution 3 observations đại diện
st.subheader("5. Phân phối predictive — 3 quan sát đại diện")
probs_all = pp["probs"]   # (M, n_test)
idx_high = int(np.argmax(pred_mean))
idx_unc  = int(np.argmin(np.abs(pred_mean - 0.5)))
idx_low  = int(np.argmin(pred_mean))

fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
for ax, idx, title in zip(axes4,
    [idx_high, idx_unc, idx_low],
    ["Pred cao (confident +)", "Uncertain (gần 0.5)", "Pred thấp (confident −)"]):
    samp = probs_all[:, idx]
    ax.hist(samp, bins=50, density=True, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(pred_mean[idx], color="red", lw=2, label=f"Mean={pred_mean[idx]:.3f}")
    ax.axvline(pp["ci_lower"][idx], color="orange", lw=1.5, ls="--")
    ax.axvline(pp["ci_upper"][idx], color="orange", lw=1.5, ls="--",
               label=f"CI=[{pp['ci_lower'][idx]:.3f},{pp['ci_upper'][idx]:.3f}]")
    ax.set_title(f"{title}\nVar={pred_var[idx]:.5f}  Actual={y_test[idx]}", fontsize=9)
    ax.set_xlabel("P(y=1 | x, θ)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig4); plt.close(fig4)

st.success(
    f"**Prior sensitivity:** thử thay đổi σ từ 0.1 đến 10 ở Settings → "
    "posterior variance sẽ thay đổi theo, minh chứng prior ảnh hưởng đến posterior."
)

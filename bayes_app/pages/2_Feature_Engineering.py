"""
pages/2_Feature_Engineering.py — Trang 2: Feature Engineering
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="2 · Feature Engineering", page_icon="🔧", layout="wide")
st.title("Trang 2 — Feature Engineering")
st.caption("9 features từ OHLCV · công thức · phân phối · correlation")

feat, src = L.features_or_none()
if feat is None:
    st.error(src); st.stop()

df_raw, _ = L.get_raw_df()

with st.expander("⚙️ Settings", expanded=False):
    selected_features = st.multiselect(
        "Chọn feature để xem phân phối",
        options=P.FEATURE_COLS,
        default=P.FEATURE_COLS,
    )
    show_by_label = st.checkbox("So sánh phân phối theo Label (0/1)", value=True)

# ── 1. Công thức ─────────────────────────────────────────────────────────────
st.subheader("1. Công thức các features")
st.markdown(r"""
| Feature | Công thức | Ý nghĩa |
|---|---|---|
| **Return** | $r_t = \log(C_t / C_{t-1})$ | Log return ngày t |
| **Volatility** | $\sqrt{\frac{1}{10}\sum_{i=0}^{9}(r_{t-i}-\bar r)^2}$ | Biến động 10 phiên |
| **MA\_Ratio** | $MA_{10} / MA_{30}$ | Xu hướng ngắn/dài hạn |
| **Lag1** | $r_{t-1}$ | Return ngày hôm qua |
| **Lag2** | $r_{t-2}$ | Return 2 ngày trước |
| **Intraday\_Range** | $(H_t - L_t) / C_t$ | Biên độ dao động ngày |
| **Close\_Position** | $(C_t - L_t) / (H_t - L_t)$ | Vị trí đóng cửa trong ngày (0=thấp, 1=cao) |
| **Volume\_Ratio** | $V_t / MA_{20}(V)$ | Khối lượng so với trung bình 20 phiên |
| **Volatility\_Ratio** | $Vol_{10} / Vol_{30}$ | Biến động ngắn/dài hạn |

**Label động (no leakage):**
$$\text{Label}_t = \mathbb{1}\!\left[|r_{t+1}| > \tau_t\right], \quad \tau_t = Q_{0.80}\!\left(|r_{t-126}|,\dots,|r_{t-1}|\right)$$
""")

# ── 2. Thống kê mô tả ─────────────────────────────────────────────────────────
st.subheader("2. Thống kê mô tả")
st.dataframe(feat[P.FEATURE_COLS].describe().round(4), use_container_width=True)

# ── 3. Phân phối ─────────────────────────────────────────────────────────────
st.subheader("3. Phân phối từng feature")
if not selected_features:
    st.warning("Chọn ít nhất 1 feature.")
else:
    n_feat = len(selected_features)
    ncols = min(3, n_feat)
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, feat_name in enumerate(selected_features):
        ax = axes_flat[i]
        if show_by_label:
            for lbl, color, name in [(0, "steelblue", "Normal"), (1, "tomato", "Strong Mov.")]:
                data = feat.loc[feat["Label"] == lbl, feat_name].dropna()
                ax.hist(data, bins=50, alpha=0.55, color=color, density=True, label=name)
            ax.legend(fontsize=7)
        else:
            ax.hist(feat[feat_name].dropna(), bins=60, color="steelblue", alpha=0.8, density=True)
        ax.set_title(feat_name, fontsize=10)
        ax.grid(alpha=0.3)

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── 4. Timeline 4 key features ───────────────────────────────────────────────
st.subheader("4. Timeline — 4 feature theo thời gian")
key4 = ["Return", "Volatility", "MA_Ratio", "Volume_Ratio"]
fig2, axes2 = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
colors4 = ["steelblue", "seagreen", "coral", "mediumpurple"]
for ax, fname, col in zip(axes2, key4, colors4):
    ax.plot(feat.index, feat[fname], lw=0.7, color=col, alpha=0.85)
    ax.set_title(fname, fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2); plt.close(fig2)

# ── 5. Correlation heatmap ────────────────────────────────────────────────────
st.subheader("5. Correlation heatmap")
corr = feat[P.FEATURE_COLS + ["Label"]].corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
import matplotlib.colors as mcolors
im = ax3.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax3)
labels = corr.columns.tolist()
ax3.set_xticks(range(len(labels))); ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax3.set_yticks(range(len(labels))); ax3.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        if i >= j:
            ax3.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(corr.values[i,j]) > 0.5 else "black")
ax3.set_title("Feature Correlation Matrix", fontsize=12)
plt.tight_layout()
st.pyplot(fig3); plt.close(fig3)

st.markdown(
    "**Lưu ý:** `Volatility` và `Volatility_Ratio` tương quan cao với nhau "
    "(cùng đo biến động). `Lag1`/`Lag2` tương quan cao với `Return` vì cùng là "
    "log return ở lag khác nhau — bình thường trong time-series."
)

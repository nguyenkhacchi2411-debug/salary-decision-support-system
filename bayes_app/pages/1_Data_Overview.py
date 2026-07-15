"""
pages/1_Data_Overview.py — Trang 1: Data Overview
==========================================================================
Thống kê mô tả · Tổng quan thị trường · Phân phối return · Rolling volatility ·
Class balance của nhãn động.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import loaders as L
from core import pipeline as P

st.set_page_config(page_title="1 · Data Overview", page_icon="📈", layout="wide")
st.title("Trang 1 — Data Overview")
st.caption("Tổng quan dữ liệu S&P 500 thô và nhãn động")

# --------------------------------------------------------------------------- #
# Lấy dữ liệu
# --------------------------------------------------------------------------- #
df, src = L.get_raw_df()
if df is None:
    st.error(src)
    st.info("Hãy nạp dữ liệu ở **Trang chủ** trước.")
    st.stop()

feat, _ = L.features_or_none()
q = P.data_quality_report(df)

with st.expander("⚙️ Settings", expanded=False):
    vol_window = st.slider(
        "Cửa sổ rolling volatility (phiên) — chỉ ảnh hưởng biểu đồ ở mục 4",
        min_value=5, max_value=60, value=10, step=1,
        help="Trang này dùng 10 phiên cho feature 'Volatility'. Slider này chỉ đổi cách "
             "minh hoạ, không đổi feature dùng để train.",
    )
    show_table = st.checkbox("Hiện bảng dữ liệu thô (20 dòng đầu)", value=False)

st.caption(f"Nguồn: **{src}** · {q['period'][0]} → {q['period'][1]} · {q['shape'][0]:,} phiên")

# --------------------------------------------------------------------------- #
# 1. Quality check + thống kê mô tả
# --------------------------------------------------------------------------- #
st.subheader("1. Kiểm tra chất lượng & thống kê mô tả")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Số phiên", f"{q['actual_days']:,}")
c2.metric("Missing values", q["missing_values"])
c3.metric("Inf values", q["inf_values"])
c4.metric("Ngày nghỉ lễ", q["missing_bdays"], help="business days không giao dịch")
c5.metric("Sorted theo thời gian", "✓" if q["is_sorted"] else "✗")

st.dataframe(df.describe().round(2), use_container_width=True)
if show_table:
    st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------------------------------- #
# 2. Tổng quan thị trường (Close / Volume / Log return)
# --------------------------------------------------------------------------- #
st.subheader("2. Tổng quan thị trường")

daily_ret = q["daily_return"]

fig, axes = plt.subplots(3, 1, figsize=(13, 9))

axes[0].plot(df.index, df["Close"], lw=1.0, color="steelblue")
axes[0].set_title("Close Price")
axes[0].set_ylabel("Price (USD)")
axes[0].grid(alpha=0.3)

axes[1].bar(df.index, df["Volume"], width=1.5, color="coral", alpha=0.7)
axes[1].set_title("Trading Volume")
axes[1].set_ylabel("Volume")
axes[1].grid(alpha=0.3)

axes[2].plot(daily_ret.index, daily_ret, lw=0.6, color="seagreen", alpha=0.8)
axes[2].axhline(0, color="black", lw=0.8)

axes[2].set_title("Log Return")
axes[2].set_ylabel("Log Return")
axes[2].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 3. Phân phối Log Return
# --------------------------------------------------------------------------- #
st.subheader("3. Phân phối Log Return")
strong_pct = float((daily_ret.abs() > 0.01).mean() * 100)
fig2, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(daily_ret, bins=80, color="steelblue", alpha=0.85, density=True)
ax[0].axvline(0.01, color="red", ls="--"); ax[0].axvline(-0.01, color="red", ls="--")
ax[0].set_title(f"Histogram (|r|>1%: {strong_pct:.1f}%)")
ax[0].set_xlabel("Log Return"); ax[0].grid(alpha=0.3)
ax[1].boxplot(daily_ret, vert=False, widths=0.6)
ax[1].set_title("Boxplot"); ax[1].set_xlabel("Log Return"); ax[1].grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2); plt.close(fig2)
st.markdown(
    f"- Độ lệch chuẩn return: **{daily_ret.std():.4f}** · trung bình: **{daily_ret.mean():.5f}**  \n"
    f"- Phần trăm phiên biến động mạnh (|return|>1%): **{strong_pct:.1f}%**  \n"
    "- Phân phối có đuôi nặng (leptokurtic) — đặc trưng dữ liệu tài chính, lý do dùng "
    "return thay vì giá thô."
)

# --------------------------------------------------------------------------- #
# 4. Rolling Volatility
# --------------------------------------------------------------------------- #
st.subheader(f"4. Rolling Volatility ({vol_window} phiên)")
roll_vol = daily_ret.rolling(window=vol_window).std()
fig3, ax3 = plt.subplots(figsize=(13, 4))
ax3.plot(roll_vol.index, roll_vol, lw=1.0, color="steelblue", alpha=0.85)
ax3.axhline(roll_vol.median(), color="red", ls="--", lw=1.2,
            label=f"Median = {roll_vol.median():.4f}")
ax3.set_title(f"{vol_window}-Day Rolling Volatility")
ax3.set_ylabel("Std of Log Return"); ax3.legend(); ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3); plt.close(fig3)
st.caption(
    "Volatility clustering: các giai đoạn biến động cao đi liền nhau (vd. khủng "
    "hoảng) — đây là tín hiệu mà mô hình uncertainty kỳ vọng phản ánh được."
)

# --------------------------------------------------------------------------- #
# 5. Class Balance của nhãn động
# --------------------------------------------------------------------------- #
st.subheader("5. Class balance — nhãn động (rolling-quantile, no leakage)")
st.markdown(
    r"""
$$\text{Label}_t = \mathbb{1}\!\left[\,|r_{t+1}| > \tau_t\,\right], \qquad
\tau_t = Q_{0.80}\big(|r_{t-126}|,\dots,|r_{t-1}|\big)$$

Nhãn = 1 nếu biến động phiên **kế tiếp** vượt ngưỡng top-20% của 126 phiên gần
nhất. Ngưỡng động thích nghi theo chế độ thị trường; `shift(1)`/`shift(-1)` đảm
bảo không rò rỉ tương lai.
"""
)
counts = feat["Label"].value_counts().sort_index()
n0 = int(counts.get(0, 0)); n1 = int(counts.get(1, 0)); ntot = n0 + n1
cc1, cc2, cc3 = st.columns(3)
cc1.metric("Label 0 — Normal", f"{n0:,}", f"{n0/ntot*100:.1f}%")
cc2.metric("Label 1 — Strong Movement", f"{n1:,}", f"{n1/ntot*100:.1f}%")
cc3.metric("Imbalance ratio", f"{(n0/max(n1,1)):.2f} : 1")

fig4, ax4 = plt.subplots(figsize=(5, 3.2))
ax4.bar(["Normal (0)", "Strong (1)"], [n0, n1],
        color=["steelblue", "tomato"], alpha=0.85)
for i, v in enumerate([n0, n1]):
    ax4.text(i, v, f"{v}\n{v/ntot*100:.1f}%", ha="center", va="bottom", fontsize=9)
ax4.set_ylabel("Số quan sát"); ax4.set_title("Phân bố nhãn"); ax4.grid(alpha=0.3, axis="y")
plt.tight_layout()
st.pyplot(fig4); plt.close(fig4)
st.info(
    "Mất cân bằng ~80/20 là lý do toàn pipeline ưu tiên **F2-score** và "
    "**calibration** thay vì accuracy thuần (xem Trang 3–6)."
)

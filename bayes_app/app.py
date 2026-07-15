"""
app.py — Trang chủ
==================
Hệ thống đánh giá độ tin cậy dữ liệu cho pipeline phân tích tài chính.
Bayesian Uncertainty-Aware Financial Risk Forecasting.

Chạy:  streamlit run app.py
"""
import os
import streamlit as st

from core import loaders as L
from core import pipeline as P

st.set_page_config(
    page_title="Bayesian Risk Forecasting",
    page_icon="📊",
    layout="wide",
)

st.title("Hệ thống đánh giá độ tin cậy dữ liệu — Bayesian Risk Forecasting")
st.caption(
    "Bayesian Logistic Regression · Posterior Predictive · Uncertainty Quantification · "
    "Confidence Score — S&P 500 (2015–2024)"
)

st.markdown(
    """
Mục tiêu của hệ thống **không phải tối đa hóa accuracy**, mà là **định lượng độ
tin cậy của từng dự đoán**: mô hình Bayesian cho ra cả *phân phối* xác suất dự
đoán (posterior predictive), từ đó suy ra *uncertainty* và *confidence score* —
nền tảng cho việc lọc/đánh giá độ tin cậy dữ liệu trong pipeline tài chính.

Toàn bộ app **tái hiện trực tiếp từ dữ liệu thô** (không nạp model `.pkl`): mọi
hyperparameter em chỉnh sẽ kích hoạt tính lại pipeline, nên các trang phản ánh
đúng tác động của lựa chọn mô hình.
"""
)

# --------------------------------------------------------------------------- #
# Trạng thái dữ liệu
# --------------------------------------------------------------------------- #
st.subheader("Nguồn dữ liệu")

with st.expander("⚙️ Settings — Nguồn dữ liệu", expanded=False):
    st.markdown(
        f"""
App tìm dữ liệu theo thứ tự:
1. File em **upload trong phiên** (bên dưới), hoặc
2. `data/raw/sp500.csv` cạnh thư mục app (đường dẫn: `{L.DEFAULT_RAW_PATH}`).

Định dạng chấp nhận: file OHLCV gốc của yfinance (header 2 dòng MultiIndex như
NB1) **hoặc** file single-header có cột `Close, High, Low, Open, Volume`.
"""
    )
    uploaded = st.file_uploader("Upload sp500.csv (tuỳ chọn)", type=["csv"])
    if uploaded is not None:
        st.session_state["uploaded_raw"] = uploaded.getvalue()
        st.success("Đã nạp file upload cho phiên này.")
    if st.session_state.get("uploaded_raw") is not None:
        if st.button("Xoá file upload, dùng lại data/raw/sp500.csv"):
            st.session_state.pop("uploaded_raw", None)
            st.rerun()

df, src = L.get_raw_df()
if df is None:
    st.error(src)
    st.info(
        "Gợi ý: tạo thư mục `data/raw/` cạnh `app.py` và đặt `sp500.csv` vào đó, "
        "hoặc upload file ở mục Settings phía trên."
    )
    st.stop()

q = P.data_quality_report(df)
c1, c2, c3 = st.columns(3)
c1.metric("Số phiên", f"{q['shape'][0]:,}")
c2.metric("Khoảng thời gian", f"{q['period'][0].year}–{q['period'][1].year}")
c3.metric("Missing values", q["missing_values"])
st.caption(f"Nguồn đang dùng: **{src}** · {q['period'][0]} → {q['period'][1]}")

st.divider()
st.subheader("Các trang phân tích")
st.markdown(
    """
| Trang | Nội dung |
|---|---|
| **1 · Data Overview** | Thống kê mô tả, phân phối return, rolling volatility, class balance | 
| **2 · Feature Engineering** | 9 feature, công thức, correlation heatmap |
| **3 · Baseline Logistic Regression** | ROC, confusion matrix, threshold (C, max_iter) |
| **4 · Bayesian Logistic Regression** ⭐ | Posterior, posterior predictive, uncertainty (prior σ, M, CI) |
| **5 · Confidence & Threshold** | Confidence score, accuracy theo bucket, threshold slider | 
| **6 · LR vs BLR** | ROC chồng, bảng so sánh, calibration | 
| **7 · Uncertainty Impact** | Uncertainty ↔ error, theo thời gian, theo regime | 

Chọn trang ở thanh điều hướng bên trái để bắt đầu.
"""
)

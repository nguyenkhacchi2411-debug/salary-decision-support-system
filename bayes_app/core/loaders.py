"""
core/loaders.py
===============
Lớp truy cập dữ liệu DÙNG CHUNG cho cả 7 trang, có st.cache_data để tránh
tính lại các bước nặng (build features, train LR, BLR online update, posterior
predictive). Mọi trang import từ đây thay vì gọi pipeline trực tiếp.

Nguồn dữ liệu (đã thống nhất: tái hiện từ raw, không .pkl):
  1) file người dùng upload trong phiên (st.session_state['uploaded_raw']), hoặc
  2) data/raw/sp500.csv cạnh thư mục app.
"""
from __future__ import annotations
import os
import io
import numpy as np
import pandas as pd
import streamlit as st

from core import pipeline as P

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_PATH = os.path.join(APP_DIR, "data", "raw", "sp500.csv")


# --------------------------------------------------------------------------- #
# Raw data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_raw_from_path(path: str, mtime: float) -> pd.DataFrame:
    # mtime nằm trong chữ ký để cache tự refresh khi file thay đổi
    return P.load_raw_ohlcv(path)


@st.cache_data(show_spinner=False)
def _load_raw_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(raw_bytes)
    # ghi tạm rồi đọc bằng cùng loader để dùng chung logic MultiIndex
    tmp = os.path.join(APP_DIR, "data", "raw", "_uploaded_tmp.csv")
    with open(tmp, "wb") as f:
        f.write(raw_bytes)
    return P.load_raw_ohlcv(tmp)


def get_raw_df():
    """Trả về (df_raw, source_label) hoặc (None, message) nếu chưa có dữ liệu."""
    up = st.session_state.get("uploaded_raw")
    if up is not None:
        try:
            return _load_raw_from_bytes(up), "file upload trong phiên"
        except Exception as e:
            return None, f"Lỗi đọc file upload: {e}"
    if os.path.exists(DEFAULT_RAW_PATH):
        try:
            mt = os.path.getmtime(DEFAULT_RAW_PATH)
            return _load_raw_from_path(DEFAULT_RAW_PATH, mt), "data/raw/sp500.csv"
        except Exception as e:
            return None, f"Lỗi đọc data/raw/sp500.csv: {e}"
    return None, "Chưa có dữ liệu. Hãy đặt sp500.csv vào data/raw/ hoặc upload ở trang chủ."


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_features(raw_key: str) -> pd.DataFrame:
    df, _ = get_raw_df()
    if df is None:
        raise RuntimeError("Không có dữ liệu raw.")
    return P.build_features(df)


def features_or_none():
    df, src = get_raw_df()
    if df is None:
        return None, src
    # raw_key: dùng shape+period làm khóa cache đơn giản
    raw_key = f"{df.shape}-{df.index.min()}-{df.index.max()}"
    return get_features(raw_key), src


# --------------------------------------------------------------------------- #
# Split + scaler + Logistic Regression (NB3)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_lr_bundle(feat_key: str, C: float, max_iter: int):
    feat, _ = features_or_none()
    sp = P.chronological_split(feat)
    scaler = P.fit_scaler(sp["X_train"])
    Xtr = scaler.transform(sp["X_train"])
    Xval = scaler.transform(sp["X_val"])
    Xte = scaler.transform(sp["X_test"])
    model = P.train_logistic(Xtr, sp["y_train"], C=C, max_iter=max_iter)
    val_prob = model.predict_proba(Xval)[:, 1]
    test_prob = model.predict_proba(Xte)[:, 1]
    return {
        "split": sp, "scaler": scaler,
        "X_train_s": Xtr, "X_val_s": Xval, "X_test_s": Xte,
        "model": model, "coef": model.coef_[0], "intercept": model.intercept_[0],
        "val_prob": val_prob, "test_prob": test_prob,
    }


# --------------------------------------------------------------------------- #
# Bayesian Logistic Regression (NB4)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_blr_bundle(feat_key: str, prior_sigma: float, C: float, max_iter: int,
                   batch_size: int = P.DEFAULT_BATCH_SIZE,
                   discount: float = P.DEFAULT_DISCOUNT):
    """
    BLR dùng CHUNG scaler với LR (fit trên train). Trả về posterior + augmented X.
    """
    lr = get_lr_bundle(feat_key, C, max_iter)
    sp = lr["split"]
    Xtr_aug = P.add_intercept(lr["X_train_s"])
    Xval_aug = P.add_intercept(lr["X_val_s"])
    Xte_aug = P.add_intercept(lr["X_test_s"])
    n_params = Xtr_aug.shape[1]
    pm, pc, pv = P.init_prior(sp["y_train"], n_params, prior_sigma=prior_sigma)
    post = P.bayesian_online_update(
        Xtr_aug, sp["y_train"], pm, pc,
        batch_size=batch_size, discount=discount,
    )
    return {
        "split": sp, "prior_mean": pm, "prior_cov": pc, "prior_var": pv,
        "posterior": post,
        "X_train_aug": Xtr_aug, "X_val_aug": Xval_aug, "X_test_aug": Xte_aug,
    }


def feat_cache_key():
    """Khóa cache ổn định cho dataset hiện tại."""
    df, _ = get_raw_df()
    if df is None:
        return None
    return f"{df.shape}-{df.index.min()}-{df.index.max()}"

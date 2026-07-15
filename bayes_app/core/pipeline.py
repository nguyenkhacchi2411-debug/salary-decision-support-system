"""
core/pipeline.py
================
Tái hiện trung thực toàn bộ pipeline từ 7 notebooks (NB1 -> NB7), KHÔNG dùng .pkl.
Mọi trang Streamlit gọi chung module này. Các hàm nặng được cache ở tầng trang
(st.cache_data) — ở đây giữ thuần numpy/pandas/sklearn để dễ test & tái sử dụng.

Quy ước quan trọng (đồng bộ với notebook):
  - FEATURE_COLS: đúng 9 feature cuối cùng của NB2.
  - Label động (rolling-quantile, no leakage) y hệt NB2.
  - Split 60/20/20 chronological (NB3..NB7).
  - StandardScaler fit CHỈ trên train (NB3).
  - BLR = Bayesian Online Update + Laplace Approximation + discount (NB4).
  - Posterior predictive = Monte Carlo từ N(mean, cov) (NB4/NB6).
  - Confidence = 1 - robust_quantile_norm(CI_Width) (NB5/NB6).

LƯU Ý CHỦ ĐÍCH (đã thống nhất với giảng viên hướng dẫn):
  Trong bayesian_online_update, bước MAP dùng prior-precision DẠNG ĐƯỜNG CHÉO
  (np.diag(cur_cov)) còn bước Hessian dùng nghịch đảo ma trận ĐẦY ĐỦ. Đây là
  điểm bất nhất nhỏ vốn có trong NB4; ta GIỮ NGUYÊN để app khớp số với notebook,
  không tự ý sửa.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, brier_score_loss,
)

# --------------------------------------------------------------------------- #
# Hằng số đồng bộ với notebook
# --------------------------------------------------------------------------- #
FEATURE_COLS = [
    "Return", "Volatility", "MA_Ratio", "Lag1", "Lag2",
    "Intraday_Range", "Close_Position", "Volume_Ratio", "Volatility_Ratio",
]
PARAM_LABELS = ["intercept"] + FEATURE_COLS

LABEL_WINDOW = 126        # NB2: ~6 tháng giao dịch
LABEL_QUANTILE = 0.80     # NB2: top 20% biến động mạnh

# Mặc định BLR (NB4)
DEFAULT_PRIOR_SIGMA = 1.0   # slider 0.1..10 ; prior_variance = sigma**2
DEFAULT_BATCH_SIZE = 20
DEFAULT_DISCOUNT = 0.95
DEFAULT_MAP_LR = 0.01
DEFAULT_MAP_NITER = 300
DEFAULT_MC_SAMPLES = 1000


# --------------------------------------------------------------------------- #
# NB1 — Load raw OHLCV
# --------------------------------------------------------------------------- #
def load_raw_ohlcv(path: str) -> pd.DataFrame:
    """
    Đọc file S&P 500 gốc. Hỗ trợ 2 định dạng:
      (a) MultiIndex 2-dòng header (yfinance, như NB1): header=[0,1]
      (b) Single header đã sạch: Close/High/Low/Open/Volume

    Trả về DataFrame OHLCV với DatetimeIndex đã sort, cột chuẩn:
      ['Close', 'High', 'Low', 'Open', 'Volume']
    """
    # Thử đọc dạng MultiIndex trước (giống NB1)
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            # Lấy level 0 (Price) làm tên cột — giống "df.columns = ['Close',...]"
            level0 = [c[0] for c in df.columns]
            df.columns = level0
    except Exception:
        df = pd.read_csv(path, index_col=0)

    # Chuẩn hóa tên cột về Title-case nếu cần
    rename_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=rename_map)

    keep = ["Close", "High", "Low", "Open", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"File thiếu cột {missing}. Cột hiện có: {list(df.columns)}. "
            "Cần OHLCV chuẩn (Close/High/Low/Open/Volume)."
        )
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    # Ép kiểu số (phòng trường hợp đọc ra string)
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill()          # NB1: forward-fill khoảng trống nhỏ
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Bản kiểm tra chất lượng dữ liệu (NB1 mục 4)."""
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
    missing_days = bdays.difference(df.index)
    daily_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return {
        "shape": df.shape,
        "period": (df.index.min().date(), df.index.max().date()),
        "missing_values": int(df.isnull().sum().sum()),
        "expected_bdays": int(len(bdays)),
        "actual_days": int(len(df)),
        "missing_bdays": int(len(missing_days)),
        "inf_values": int(np.isinf(df.select_dtypes("number")).sum().sum()),
        "is_sorted": bool(df.index.is_monotonic_increasing),
        "strong_move_pct": float((daily_ret.abs() > 0.01).mean() * 100),
        "daily_return": daily_ret,
    }


# --------------------------------------------------------------------------- #
# NB2 — Feature engineering + dynamic label
# --------------------------------------------------------------------------- #
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo 9 feature + Label động y hệt NB2. Trả về DataFrame (FEATURE_COLS + Label),
    đã dropna — tương đương sp500_features.csv.
    """
    df = df_raw.copy()

    # 3.1 Log Return
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    # 3.2 Rolling Volatility (10)
    df["Volatility"] = df["Return"].rolling(window=10).std()
    # 3.3 MA & MA_Ratio
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA_Ratio"] = df["MA10"] / df["MA30"]
    # 3.4 Lag
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    # 3.5 Intraday Range
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Close"]
    # 3.6 Close Position
    hl = df["High"] - df["Low"]
    df["Close_Position"] = np.where(hl > 0, (df["Close"] - df["Low"]) / hl, 0.5)
    # Volume Ratio
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA20"] + 1e-8)
    # Volatility Ratio
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volatility_30"] = df["Return"].rolling(30).std()
    df["Volatility_Ratio"] = df["Volatility_10"] / (df["Volatility_30"] + 1e-8)

    # 4. Dynamic Label (rolling quantile, no leakage)
    df["Threshold"] = (
        df["Return"].abs()
        .shift(1)
        .rolling(window=LABEL_WINDOW, min_periods=LABEL_WINDOW)
        .quantile(LABEL_QUANTILE)
    )
    df["Label"] = (df["Return"].shift(-1).abs() > df["Threshold"]).astype(int)
    df = df.dropna(subset=["Threshold"])

    out = df[FEATURE_COLS + ["Label"]].copy()
    out.index.name = "Date"
    out = out.dropna()
    return out


# --------------------------------------------------------------------------- #
# NB3 — Split + Scaler + Logistic Regression
# --------------------------------------------------------------------------- #
def chronological_split(feat: pd.DataFrame, ratios=(0.60, 0.80)):
    """Split 60/20/20 theo thời gian (KHÔNG shuffle). Trả về dict các phần."""
    X = feat[FEATURE_COLS].values
    y = feat["Label"].values
    n = len(feat)
    train_end = int(n * ratios[0])
    val_end = int(n * ratios[1])
    idx = feat.index
    return {
        "X_train": X[:train_end], "y_train": y[:train_end],
        "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
        "X_test": X[val_end:], "y_test": y[val_end:],
        "dates_train": idx[:train_end],
        "dates_val": idx[train_end:val_end],
        "dates_test": idx[val_end:],
        "train_end": train_end, "val_end": val_end, "n": n,
    }


def fit_scaler(X_train):
    sc = StandardScaler()
    sc.fit(X_train)
    return sc


def train_logistic(X_train_s, y_train, C=1.0, max_iter=5000):
    """LR baseline (NB3): class_weight='balanced', solver='lbfgs'."""
    model = LogisticRegression(
        C=C, max_iter=max_iter, class_weight="balanced", solver="lbfgs",
    )
    model.fit(X_train_s, y_train)
    return model


def optimize_threshold_f2(y_true, y_prob, lo=0.10, hi=0.90, step=0.01):
    """Tối ưu ngưỡng theo F2 (beta=2) — NB3/NB5."""
    thresholds = np.arange(lo, hi + 1e-9, step)
    f2s, f1s, precs, recs = [], [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f2s.append(fbeta_score(y_true, yp, beta=2, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
    f2s = np.array(f2s)
    best = int(np.argmax(f2s))
    return {
        "thresholds": thresholds, "f2": f2s, "f1": np.array(f1s),
        "precision": np.array(precs), "recall": np.array(recs),
        "best_threshold": float(thresholds[best]),
        "best_f2": float(f2s[best]), "best_idx": best,
    }


def classification_metrics(y_true, y_prob, threshold):
    """Bộ metric chuẩn dùng cho cả LR và BLR."""
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }
    return out


# --------------------------------------------------------------------------- #
# NB4 — Bayesian Online Update + Laplace Approximation
# --------------------------------------------------------------------------- #
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X):
    """Augmented design matrix: cột intercept = 1 ở đầu (NB4)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def init_prior(y_train, n_params, prior_sigma=DEFAULT_PRIOR_SIGMA):
    """
    Prior N(mu0, Sigma0) (NB4 Phase 1):
      - intercept = log-odds class balance (class-aware)
      - feature mean = 0
      - Sigma0 = sigma^2 * I  (sigma = prior_sigma là 'prior scale')
    """
    p_pos = float(np.mean(y_train))
    p_pos = min(max(p_pos, 1e-6), 1 - 1e-6)
    prior_mean = np.zeros(n_params)
    prior_mean[0] = np.log(p_pos / (1.0 - p_pos))
    prior_var = prior_sigma ** 2
    prior_cov = np.diag(np.full(n_params, prior_var))
    return prior_mean, prior_cov, prior_var


def bayesian_online_update(
    X_aug, y, prior_mean, prior_cov,
    batch_size=DEFAULT_BATCH_SIZE, discount=DEFAULT_DISCOUNT,
    lr=DEFAULT_MAP_LR, n_iter=DEFAULT_MAP_NITER,
):
    """
    Bayesian Online Update với Laplace Approximation (NB4 Phase 2).
    Mỗi batch (theo thứ tự thời gian):
      1. Discount:  Sigma' = Sigma / lambda          (forgetting -> inflate variance)
      2. MAP:       gradient ascent trên log-posterior
                    grad = X'(y - p)  -  (theta - mu)/diag(Sigma')   [prior dạng đường chéo]
      3. Hessian:   H = X'WX + inv(Sigma'),  W = diag(p(1-p))         [prior đầy đủ]
      4. Laplace:   Sigma <- inv(H)
    Trả về mean/cov cuối + lịch sử để vẽ posterior evolution.
    """
    cur_mean = prior_mean.copy().astype(float)
    cur_cov = prior_cov.copy().astype(float)
    mean_hist, cov_hist, grad_norms = [], [], []

    n_total = len(X_aug)
    n_batches = int(np.ceil(n_total / batch_size))

    for b in range(n_batches):
        start, end = b * batch_size, min((b + 1) * batch_size, n_total)
        X_b, y_b = X_aug[start:end], y[start:end]

        # Step 1: discount
        cur_cov = cur_cov / discount
        prior_var_d = np.diag(cur_cov)               # prior đường chéo cho MAP

        # Step 2: MAP via gradient ascent
        theta = cur_mean.copy()
        grad = np.zeros_like(theta)
        for _ in range(n_iter):
            p_hat = sigmoid(X_b @ theta)
            grad_ll = X_b.T @ (y_b - p_hat)
            grad_lp = -(theta - cur_mean) / (prior_var_d + 1e-10)
            grad = grad_ll + grad_lp
            theta = theta + lr * grad
        theta_map = theta
        grad_norms.append(float(np.linalg.norm(grad)))

        # Step 3: Hessian tại theta_map (prior precision đầy đủ)
        p_map = sigmoid(X_b @ theta_map)
        W = p_map * (1.0 - p_map)
        H = (X_b.T * W) @ X_b + np.linalg.inv(cur_cov)

        # Step 4: Laplace covariance
        cur_cov = np.linalg.inv(H)
        cur_mean = theta_map

        mean_hist.append(cur_mean.copy())
        cov_hist.append(cur_cov.copy())

    return {
        "mean": cur_mean, "cov": cur_cov,
        "mean_history": np.array(mean_hist),
        "cov_history": cov_hist,
        "var_history": np.array([np.diag(c) for c in cov_hist]),
        "grad_norms": np.array(grad_norms),
        "n_batches": n_batches,
    }


# --------------------------------------------------------------------------- #
# NB4/NB6 — Posterior Predictive (Monte Carlo)
# --------------------------------------------------------------------------- #
def posterior_predictive(mean, cov, X_aug, M=DEFAULT_MC_SAMPLES,
                         cred_level=0.95, seed=42):
    """
    Monte Carlo posterior predictive (NB4 Phase 4):
      theta^(m) ~ N(mean, cov);  pi_m = sigmoid(X theta^(m))
      pred_mean, pred_var, credible interval [alpha/2, 1-alpha/2], CI width.
    cred_level: 0.80 / 0.90 / 0.95 / 0.99 (slider trang 4).
    """
    rng = np.random.default_rng(seed)
    theta_samples = rng.multivariate_normal(mean=mean, cov=cov, size=M)  # (M, p)
    logits = theta_samples @ X_aug.T                                     # (M, n)
    probs = sigmoid(logits)

    alpha = 1.0 - cred_level
    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    pred_mean = probs.mean(axis=0)
    pred_var = probs.var(axis=0)
    ci_lower = np.percentile(probs, lo_pct, axis=0)
    ci_upper = np.percentile(probs, hi_pct, axis=0)
    ci_width = ci_upper - ci_lower
    return {
        "probs": probs, "pred_mean": pred_mean, "pred_var": pred_var,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "ci_width": ci_width,
        "cred_level": cred_level,
    }


# --------------------------------------------------------------------------- #
# NB5/NB6 — Confidence Score
# --------------------------------------------------------------------------- #
def confidence_from_ciwidth(ci_width):
    """
    Robust quantile normalization (NB5):
      U_hat = clip((U - p2.5)/(p97.5 - p2.5), 0, 1);  Confidence = 1 - U_hat.
    Trả về (confidence, info_dict).
    """
    p_lo = np.percentile(ci_width, 2.5)
    p_hi = np.percentile(ci_width, 97.5)
    denom = (p_hi - p_lo) if (p_hi - p_lo) > 1e-12 else 1e-12
    u_hat = np.clip((ci_width - p_lo) / denom, 0, 1)
    confidence = 1.0 - u_hat
    return confidence, {"p2.5": float(p_lo), "p97.5": float(p_hi)}


def confidence_buckets(confidence):
    """Chia tertile data-driven (NB5): High>=p67, Low<p33, còn lại Medium."""
    p33 = np.percentile(confidence, 33)
    p67 = np.percentile(confidence, 67)
    buckets = np.where(confidence >= p67, "High",
                       np.where(confidence < p33, "Low", "Medium"))
    return buckets, {"p33": float(p33), "p67": float(p67)}

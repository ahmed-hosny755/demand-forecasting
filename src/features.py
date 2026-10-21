"""Leakage-safe feature engineering, shared by training and serving.

Rule: a forecast is made on day t for day t + HORIZON, so every history-based
feature may only use data from day t or earlier (i.e. >= HORIZON days before the target day).
"""
import numpy as np
import pandas as pd

from src import config as cfg

RAW_COLUMNS = ["date", "store_id", "product_id", "category", "demand",
               "price", "discount", "promotion", "epidemic"]


# ── Loading and checks ────────────────────────────────────────
def load_clean_data(path=cfg.DATA_PATH) -> pd.DataFrame:
    """Read the cleaned CSV produced by notebook 01 and add a per-series id."""
    df = pd.read_csv(path, parse_dates=["date"], usecols=lambda c: c in set(RAW_COLUMNS))
    df["series_id"] = df["store_id"].astype(str) + "_" + df["product_id"].astype(str)
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def assert_complete_daily_panel(df: pd.DataFrame) -> None:
    """shift()-based lags are only correct if every series has one row per consecutive day."""
    gaps = df.sort_values(["series_id", "date"]).groupby("series_id")["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("Each series must contain consecutive daily rows (found gaps or duplicates).")


# ── Feature builders (training) ───────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["dayofweek"] = out["date"].dt.dayofweek
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    return out


def add_history_features(df: pd.DataFrame, horizon: int = cfg.HORIZON) -> pd.DataFrame:
    """Lag / rolling features computed per series; rolling stats use data shifted by `horizon`."""
    if min(cfg.LAG_DAYS) < horizon:
        raise ValueError(f"Every lag must be >= horizon ({horizon}); a shorter lag would leak the future.")
    out = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    g = out.groupby("series_id", sort=False)

    for lag in cfg.LAG_DAYS:
        out[f"lag_{lag}"] = g["demand"].shift(lag)
    for w in cfg.ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = g["demand"].transform(lambda s, w=w: s.shift(horizon).rolling(w).mean())
    out["roll_std_28"] = g["demand"].transform(lambda s: s.shift(horizon).rolling(28).std())
    out["epidemic_lag_7"] = g["epidemic"].shift(horizon)       # regime that was observable at forecast time
    return out


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All features for every row; the warm-up rows without full history are dropped."""
    assert_complete_daily_panel(df)
    out = add_history_features(add_calendar_features(df))
    out = out.dropna(subset=cfg.HISTORY + ["epidemic_lag_7"]).reset_index(drop=True)
    out["epidemic_lag_7"] = out["epidemic_lag_7"].astype(int)
    return out


# ── Categorical handling (must be identical in training and serving) ──
def category_levels(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].astype(str).unique()) for c in cfg.CATEGORICAL}


def apply_categories(X: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = X.copy()
    for c in cfg.CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    return X


# ── Feature builders (serving) ────────────────────────────────
def history_features_for(series_hist: pd.DataFrame, target_date, horizon: int = cfg.HORIZON) -> dict:
    """History features for ONE series and ONE target day, using only data up to the forecast origin."""
    target = pd.Timestamp(target_date)
    origin = target - pd.Timedelta(days=horizon)
    h = series_hist[series_hist["date"] <= origin].set_index("date").sort_index()   # nothing newer than the origin

    def value(col, day):
        return h[col].get(day, np.nan)

    def window(w):
        idx = pd.date_range(target - pd.Timedelta(days=horizon + w - 1), target - pd.Timedelta(days=horizon))
        return h["demand"].reindex(idx)

    feats = {f"lag_{lag}": value("demand", target - pd.Timedelta(days=lag)) for lag in cfg.LAG_DAYS}
    for w in cfg.ROLL_WINDOWS:
        s = window(w)
        feats[f"roll_mean_{w}"] = s.mean() if s.notna().all() else np.nan
    s28 = window(28)
    feats["roll_std_28"] = s28.std() if s28.notna().all() else np.nan
    feats["epidemic_lag_7"] = value("epidemic", target - pd.Timedelta(days=horizon))
    return feats


def build_inference_row(history: pd.DataFrame, store_id: str, product_id: str, target_date,
                        price: float, discount: float, promotion: int, epidemic=None) -> pd.DataFrame:
    """One-row feature frame for a single forecast request (same logic as training)."""
    sid = f"{store_id}_{product_id}"
    hist = history[history["series_id"] == sid]
    if hist.empty:
        raise KeyError(f"Unknown series {sid}")
    target = pd.Timestamp(target_date)

    row = {"store_id": store_id, "product_id": product_id, "category": hist["category"].iloc[-1],
           "price": price, "discount": discount, "promotion": int(promotion),
           "month": target.month, "dayofweek": target.dayofweek, "week": int(target.isocalendar()[1]),
           **history_features_for(hist, target)}
    if any(pd.isna(row[c]) for c in cfg.HISTORY + ["epidemic_lag_7"]):
        raise ValueError("Not enough history for this series before the forecast origin "
                         f"({target.date()} minus {cfg.HORIZON} days needs 35 days of past demand).")
    row["epidemic_lag_7"] = int(row["epidemic_lag_7"])
    if epidemic is not None:
        row["epidemic"] = int(epidemic)
    return pd.DataFrame([row])

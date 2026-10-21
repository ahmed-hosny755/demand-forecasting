"""Tests that guard against the two most dangerous forecasting bugs: data leakage and train/serve skew."""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.features import (assert_complete_daily_panel, build_inference_row, build_training_frame)

FEATURES = cfg.HISTORY + ["epidemic_lag_7"]


@pytest.fixture
def panel() -> pd.DataFrame:
    """Small synthetic panel: 2 stores x 2 products x 90 consecutive days."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=90)
    rows = []
    for store in ["S001", "S002"]:
        for product in ["P0001", "P0002"]:
            for d in dates:
                rows.append({"date": d, "store_id": store, "product_id": product, "category": "Toys",
                             "demand": int(rng.integers(20, 200)), "price": 50.0, "discount": 10,
                             "promotion": int(rng.integers(0, 2)), "epidemic": int(rng.integers(0, 2))})
    df = pd.DataFrame(rows)
    df["series_id"] = df["store_id"] + "_" + df["product_id"]
    return df


def test_lag_values_come_from_the_past(panel):
    frame = build_training_frame(panel)
    row = frame.iloc[10]
    s = panel[panel["series_id"] == row["series_id"]].set_index("date")["demand"]
    d = row["date"]
    assert row["lag_7"] == s[d - pd.Timedelta(days=7)]
    assert row["lag_28"] == s[d - pd.Timedelta(days=28)]
    assert np.isclose(row["roll_mean_7"], s[d - pd.Timedelta(days=13): d - pd.Timedelta(days=7)].mean())
    assert np.isclose(row["roll_mean_28"], s[d - pd.Timedelta(days=34): d - pd.Timedelta(days=7)].mean())


def test_features_ignore_data_newer_than_the_forecast_origin(panel):
    """Changing demand/epidemic AFTER (target - horizon) must not change the features of the target day."""
    target = pd.Timestamp("2023-03-20")
    sid = "S001_P0001"
    base = build_training_frame(panel)
    base_row = base[(base["series_id"] == sid) & (base["date"] == target)][FEATURES].iloc[0]

    altered = panel.copy()
    future = (altered["series_id"] == sid) & (altered["date"] > target - pd.Timedelta(days=cfg.HORIZON))
    altered.loc[future, "demand"] = altered.loc[future, "demand"] * 100
    altered.loc[future, "epidemic"] = 1 - altered.loc[future, "epidemic"]
    alt_row = build_training_frame(altered).query("series_id == @sid and date == @target")[FEATURES].iloc[0]

    pd.testing.assert_series_equal(base_row, alt_row)


def test_serving_features_match_training_features(panel):
    """Training/serving skew test: the single-row inference code must reproduce the training features."""
    frame = build_training_frame(panel)
    row = frame[(frame["series_id"] == "S002_P0002") & (frame["date"] == pd.Timestamp("2023-03-15"))].iloc[0]

    X = build_inference_row(panel, "S002", "P0002", row["date"], row["price"], row["discount"], row["promotion"])
    for col in FEATURES:
        assert np.isclose(X.loc[0, col], row[col]), col
    assert X.loc[0, "week"] == row["week"] and X.loc[0, "dayofweek"] == row["dayofweek"]


def test_inference_requires_enough_history(panel):
    with pytest.raises(ValueError, match="Not enough history"):
        build_inference_row(panel, "S001", "P0001", "2023-01-20", 50.0, 10, 0)


def test_unknown_series_raises(panel):
    with pytest.raises(KeyError):
        build_inference_row(panel, "S999", "P0001", "2023-03-20", 50.0, 10, 0)


def test_gaps_in_the_panel_are_rejected(panel):
    with pytest.raises(ValueError, match="consecutive"):
        assert_complete_daily_panel(panel.drop(index=50))


def test_warm_up_rows_are_dropped(panel):
    frame = build_training_frame(panel)
    assert frame[FEATURES].notna().all().all()
    assert frame["date"].min() == pd.Timestamp("2023-01-01") + pd.Timedelta(days=34)   # 28-day window + 7-day shift - 1

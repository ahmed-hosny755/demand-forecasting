"""Shared fixtures: a small synthetic panel and a ForecastService trained on it."""
import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src import config as cfg
from src.features import apply_categories, build_training_frame, category_levels
from src.serving import ForecastService


@pytest.fixture
def panel() -> pd.DataFrame:
    """Small synthetic panel: 2 stores x 2 products x 90 consecutive days (2023-01-01 → 2023-03-31)."""
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


@pytest.fixture
def service(panel, tmp_path) -> ForecastService:
    """Train two tiny models, save them like src.train does, and load them through ForecastService.load()."""
    frame = build_training_frame(panel)
    levels = category_levels(frame)
    frame = apply_categories(frame, levels)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name, features in {"epidemic_unknown": cfg.FEATURES_UNKNOWN, "epidemic_known": cfg.FEATURES_KNOWN}.items():
        model = XGBRegressor(n_estimators=20, enable_categorical=True, tree_method="hist", random_state=0)
        model.fit(frame[features], frame["demand"])
        model.save_model(model_dir / f"xgb_{name}.json")

    spec = {"horizon_days": cfg.HORIZON, "features_epidemic_unknown": cfg.FEATURES_UNKNOWN,
            "features_epidemic_known": cfg.FEATURES_KNOWN, "categories": levels, "trained_until": "2023-03-31"}
    (model_dir / "feature_spec.json").write_text(json.dumps(spec))

    history_path = tmp_path / "history.csv"
    panel.to_csv(history_path, index=False)
    return ForecastService.load(history_path=history_path, model_dir=model_dir)

"""Train the final XGBoost models (train + validation data) and evaluate them once on the test period.

Run from the project root:   python -m src.train
"""
import argparse
import json

import pandas as pd
from xgboost import XGBRegressor

from src import config as cfg
from src.evaluate import evaluate
from src.features import apply_categories, build_training_frame, category_levels, load_clean_data

SCENARIOS = {
    "epidemic_unknown": (cfg.FEATURES_UNKNOWN, "xgboost_epidemic_unknown"),
    "epidemic_known": (cfg.FEATURES_KNOWN, "xgboost_epidemic_known"),
}


def make_model(params: dict, n_estimators: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=n_estimators, eval_metric="mae", tree_method="hist",
                        enable_categorical=True, n_jobs=-1, random_state=cfg.SEED, **params)


def load_tuned_params() -> dict:
    """Best hyperparameters found in notebook 02 (reports/best_params.json), or defaults."""
    path = cfg.REPORT_DIR / "best_params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run(n_estimators: int = cfg.N_ESTIMATORS) -> dict:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_training_frame(load_clean_data())
    levels = category_levels(frame)
    frame = apply_categories(frame, levels)

    train = frame[frame["date"] < pd.Timestamp(cfg.TEST_START)]
    test = frame[frame["date"] >= pd.Timestamp(cfg.TEST_START)]
    tuned = load_tuned_params()

    results = {}
    for name, (features, key) in SCENARIOS.items():
        params = tuned.get(key, cfg.DEFAULT_PARAMS)
        model = make_model(params, n_estimators)
        model.fit(train[features], train["demand"], verbose=False)

        metrics = evaluate(test["demand"], model.predict(test[features]))
        model_path = cfg.MODEL_DIR / f"xgb_{name}.json"
        model.save_model(model_path)
        results[name] = {"params": params, "metrics": metrics, "model_path": model_path, "features": features}
        print(f"{name:18s} MAE={metrics['MAE']:.2f}  WAPE={metrics['WAPE_%']:.2f}%  Bias={metrics['Bias_%']:.2f}%")

    spec = {"horizon_days": cfg.HORIZON, "lag_days": cfg.LAG_DAYS, "roll_windows": cfg.ROLL_WINDOWS,
            "features_epidemic_unknown": cfg.FEATURES_UNKNOWN, "features_epidemic_known": cfg.FEATURES_KNOWN,
            "categories": levels, "trained_until": str(train["date"].max().date()), "n_estimators": n_estimators}
    (cfg.MODEL_DIR / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T.round(3).to_csv(cfg.REPORT_DIR / "metrics_src.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=cfg.N_ESTIMATORS)
    run(parser.parse_args().n_estimators)

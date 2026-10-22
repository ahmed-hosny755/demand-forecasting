"""Train the final XGBoost models (train + validation data), evaluate them once on the test period
and track every run with MLflow.

Run from the project root:   python -m src.train
View the runs:               mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import json
from datetime import datetime

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


def run(n_estimators: int = cfg.N_ESTIMATORS, track: bool = True) -> dict:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_training_frame(load_clean_data())
    levels = category_levels(frame)
    frame = apply_categories(frame, levels)

    train = frame[frame["date"] < pd.Timestamp(cfg.TEST_START)]
    test = frame[frame["date"] >= pd.Timestamp(cfg.TEST_START)]
    tuned = load_tuned_params()

    spec = {"horizon_days": cfg.HORIZON, "lag_days": cfg.LAG_DAYS, "roll_windows": cfg.ROLL_WINDOWS,
            "features_epidemic_unknown": cfg.FEATURES_UNKNOWN, "features_epidemic_known": cfg.FEATURES_KNOWN,
            "categories": levels, "trained_until": str(train["date"].max().date()), "n_estimators": n_estimators}
    spec_path = cfg.MODEL_DIR / "feature_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    baseline_wape = evaluate(test["demand"], test["lag_7"])["WAPE_%"]        # Naive-7 on the same test rows
    data_info = {"rows_train": len(train), "rows_test": len(test), "n_series": frame["series_id"].nunique(),
                 "train_end": str(train["date"].max().date()), "test_start": cfg.TEST_START,
                 "test_end": str(test["date"].max().date())}

    if track:
        from src import tracking
        tracking.setup_tracking()
    group = datetime.now().strftime("%Y%m%d-%H%M%S")

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

        if track:
            run_id = tracking.log_scenario(name=name, model=model, params=params, features=features,
                                           metrics=metrics, baseline_wape=baseline_wape,
                                           data_info=data_info, spec_path=spec_path, group=group)
            print(f"{'':18s} tracked in MLflow (run {run_id[:8]})")

    pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T.round(3).to_csv(cfg.REPORT_DIR / "metrics_src.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=cfg.N_ESTIMATORS)
    parser.add_argument("--no-mlflow", action="store_true", help="train without logging to MLflow")
    args = parser.parse_args()
    run(args.n_estimators, track=not args.no_mlflow)

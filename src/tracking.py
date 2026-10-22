"""MLflow helpers: one tracked run per trained scenario (params, metrics, artifacts, registered model)."""
import subprocess

import mlflow
import mlflow.xgboost

from src import config as cfg


def setup_tracking() -> None:
    """Point MLflow at the local SQLite store and make sure the experiment exists."""
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(cfg.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(cfg.MLFLOW_EXPERIMENT, artifact_location=cfg.MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)


def git_commit() -> str:
    """Short hash of the current commit, so every run can be traced back to the exact code."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cfg.ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def log_scenario(*, name, model, params, features, metrics, baseline_wape, data_info, spec_path, group="") -> str:
    """Log one trained model as an MLflow run and register it. Returns the run id."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"scenario": name, "git_commit": git_commit(), "run_group": group, "framework": "xgboost"})

        mlflow.log_params({**params, "n_estimators": model.get_params()["n_estimators"],
                           "horizon_days": cfg.HORIZON, "n_features": len(features),
                           "seed": cfg.SEED, **data_info})

        mlflow.log_metrics({
            "test_mae": float(metrics["MAE"]),
            "test_rmse": float(metrics["RMSE"]),
            "test_wape_pct": float(metrics["WAPE_%"]),
            "test_bias_pct": float(metrics["Bias_%"]),
            "baseline_naive7_wape_pct": float(baseline_wape),
            "wape_gain_vs_naive_pct": float((baseline_wape - metrics["WAPE_%"]) / baseline_wape * 100),
        })

        mlflow.log_dict({"features": list(features)}, "features.json")
        mlflow.log_artifact(str(spec_path))
        mlflow.xgboost.log_model(model, name="model",
                                 registered_model_name=f"demand-forecast-{name.replace('_', '-')}")
        return run.info.run_id

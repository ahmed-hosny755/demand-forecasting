"""The MLflow logging helper creates a run with the expected params, metrics and registered model."""
import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

from xgboost import XGBRegressor  # noqa: E402

from src import config as cfg  # noqa: E402
from src import tracking  # noqa: E402


def test_log_scenario_creates_run_with_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(cfg, "MLFLOW_ARTIFACT_DIR", tmp_path / "artifacts")
    spec = tmp_path / "feature_spec.json"
    spec.write_text("{}")

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 3)), rng.normal(size=50)
    model = XGBRegressor(n_estimators=5).fit(X, y)

    tracking.setup_tracking()
    run_id = tracking.log_scenario(name="epidemic_unknown", model=model, params={"max_depth": 3},
                                   features=["a", "b", "c"],
                                   metrics={"MAE": 8.0, "RMSE": 10.0, "WAPE_%": 9.0, "Bias_%": 0.5},
                                   baseline_wape=40.0, data_info={"rows_train": 50}, spec_path=spec)

    run = mlflow.get_run(run_id)
    assert run.data.metrics["test_wape_pct"] == 9.0
    assert np.isclose(run.data.metrics["wape_gain_vs_naive_pct"], 77.5)
    assert run.data.params["max_depth"] == "3"
    assert run.data.tags["scenario"] == "epidemic_unknown"
    assert mlflow.MlflowClient().get_registered_model("demand-forecast-epidemic-unknown") is not None

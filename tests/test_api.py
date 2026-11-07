"""API tests: request validation, error codes, and agreement with the training pipeline."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config as cfg
from src.api.main import app, get_service
from src.features import apply_categories, build_training_frame, category_levels

PAYLOAD = {"store_id": "S001", "product_id": "P0001", "target_date": "2023-04-05",
           "price": 50.0, "discount": 10, "promotion": 1}


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)                      # no `with` → startup (real model loading) is skipped
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_info_and_series(client):
    info = client.get("/info").json()
    assert info["horizon_days"] == 7 and info["n_series"] == 4 and info["history_end"] == "2023-03-31"
    assert len(client.get("/series").json()) == 4


def test_predict_returns_a_forecast(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_units"] >= 0
    assert body["model_used"] == "epidemic_unknown"
    assert body["forecast_origin"] == "2023-03-29"          # target 2023-04-05 minus 7 days
    assert body["category"] == "Toys"


def test_epidemic_input_switches_the_model(client):
    r = client.post("/predict", json={**PAYLOAD, "epidemic": 1})
    assert r.status_code == 200 and r.json()["model_used"] == "epidemic_known"


def test_unknown_series_is_404(client):
    r = client.post("/predict", json={**PAYLOAD, "store_id": "S999"})
    assert r.status_code == 404 and r.json()["detail"] == "Unknown series S999_P0001"


@pytest.mark.parametrize("bad", [{"discount": 150}, {"promotion": 2}, {"price": -5}, {"target_date": "not-a-date"}])
def test_invalid_input_is_422(client, bad):
    assert client.post("/predict", json={**PAYLOAD, **bad}).status_code == 422


@pytest.mark.parametrize("target", ["2023-01-20", "2023-06-01"])
def test_not_enough_history_is_422(client, target):
    """Too early (no 35 days of history) or too far in the future (history missing before the origin)."""
    r = client.post("/predict", json={**PAYLOAD, "target_date": target})
    assert r.status_code == 422 and "Not enough history" in r.json()["detail"]


def test_batch_matches_single_predictions(client):
    items = [PAYLOAD, {**PAYLOAD, "store_id": "S002", "product_id": "P0002"}, {**PAYLOAD, "epidemic": 0}]
    batch = client.post("/predict/batch", json={"requests": items}).json()["predictions"]
    singles = [client.post("/predict", json=i).json() for i in items]
    assert [b["forecast_units"] for b in batch] == [s["forecast_units"] for s in singles]
    assert [b["model_used"] for b in batch] == ["epidemic_unknown", "epidemic_unknown", "epidemic_known"]


def test_batch_error_names_the_failing_request(client):
    r = client.post("/predict/batch", json={"requests": [PAYLOAD, {**PAYLOAD, "store_id": "S999"}]})
    assert r.status_code == 404 and "request #1" in r.json()["detail"]


def test_api_forecast_equals_pipeline_prediction(client, service, panel):
    """End-to-end skew test: for a past day, the API must return exactly what the training pipeline's features give."""
    frame = build_training_frame(panel)
    levels = category_levels(frame)
    row = frame[(frame["series_id"] == "S002_P0002") & (frame["date"] == pd.Timestamp("2023-03-15"))]
    expected = service.models["epidemic_unknown"].predict(
        apply_categories(row, levels)[cfg.FEATURES_UNKNOWN])[0]

    r = client.post("/predict", json={"store_id": "S002", "product_id": "P0002", "target_date": "2023-03-15",
                                      "price": float(row["price"].iloc[0]), "discount": float(row["discount"].iloc[0]),
                                      "promotion": int(row["promotion"].iloc[0])})
    assert r.status_code == 200
    assert r.json()["forecast_units"] == pytest.approx(max(float(expected), 0), abs=0.01)

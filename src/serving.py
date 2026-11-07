"""Framework-independent forecasting service: loads models + history and answers forecast requests.

It reuses the exact feature code used for training (src.features), so there is no train/serve skew.
"""
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from src import config as cfg
from src.features import apply_categories, build_inference_row, load_clean_data

SCENARIO_UNKNOWN = "epidemic_unknown"
SCENARIO_KNOWN = "epidemic_known"


class ForecastService:
    def __init__(self, history: pd.DataFrame, models: dict, spec: dict):
        self.history = history
        self.models = models
        self.spec = spec
        self._by_series = {sid: g for sid, g in history.groupby("series_id")}

    # ── Loading ────────────────────────────────────────────────
    @classmethod
    def load(cls, history_path: Path = cfg.DATA_PATH, model_dir: Path = cfg.MODEL_DIR) -> "ForecastService":
        for required in (Path(history_path), Path(model_dir) / "feature_spec.json"):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run `python -m src.train` first, or mount the models and the history "
                    "file into the container (see the README).")
        spec = json.loads((Path(model_dir) / "feature_spec.json").read_text())
        models = {}
        for name in (SCENARIO_UNKNOWN, SCENARIO_KNOWN):
            model = XGBRegressor(enable_categorical=True)
            model.load_model(Path(model_dir) / f"xgb_{name}.json")
            models[name] = model
        return cls(load_clean_data(history_path), models, spec)

    # ── Information ────────────────────────────────────────────
    def info(self) -> dict:
        return {"horizon_days": self.spec["horizon_days"],
                "models_loaded": sorted(self.models),
                "trained_until": self.spec["trained_until"],
                "history_start": str(self.history["date"].min().date()),
                "history_end": str(self.history["date"].max().date()),
                "n_series": len(self._by_series)}

    def series_catalog(self) -> list:
        cat = self.history.groupby("series_id").agg(store_id=("store_id", "first"), product_id=("product_id", "first"),
                                                   category=("category", "last")).reset_index(drop=True)
        return cat.sort_values(["store_id", "product_id"]).to_dict("records")

    # ── Prediction ─────────────────────────────────────────────
    def _features(self, req: dict) -> tuple:
        """Build the one-row feature frame for a request and pick the model scenario."""
        sid = f"{req['store_id']}_{req['product_id']}"
        if sid not in self._by_series:
            raise KeyError(f"Unknown series {sid}")
        levels = self.spec["categories"]
        if req["store_id"] not in levels["store_id"] or req["product_id"] not in levels["product_id"]:
            raise ValueError(f"Series {sid} was not seen when the models were trained; retrain first.")

        X = build_inference_row(self._by_series[sid], req["store_id"], req["product_id"], req["target_date"],
                                req["price"], req["discount"], req["promotion"], req.get("epidemic"))
        scenario = SCENARIO_KNOWN if req.get("epidemic") is not None else SCENARIO_UNKNOWN
        features = self.spec[f"features_{scenario}"]
        return apply_categories(X, levels)[features], scenario, X.loc[0, "category"]

    def predict_many(self, requests: list, tag_index: bool = True) -> list:
        """Forecast several requests; rows are grouped by model so each model predicts once."""
        prepared = []
        for i, req in enumerate(requests):
            prefix = f"request #{i}: " if tag_index else ""
            try:
                prepared.append(self._features(req))
            except KeyError as e:
                raise KeyError(f"{prefix}{e.args[0]}") from None
            except ValueError as e:
                raise ValueError(f"{prefix}{e}") from None

        results = [None] * len(requests)
        for scenario, model in self.models.items():
            idx = [i for i, p in enumerate(prepared) if p[1] == scenario]
            if not idx:
                continue
            X = pd.concat([prepared[i][0] for i in idx], ignore_index=True)
            X = apply_categories(X, self.spec["categories"])        # concat can drop categorical dtype
            preds = model.predict(X)
            for i, pred in zip(idx, preds):
                req = requests[i]
                target = pd.Timestamp(req["target_date"])
                results[i] = {
                    "store_id": req["store_id"], "product_id": req["product_id"], "category": prepared[i][2],
                    "target_date": target.date(),
                    "forecast_origin": (target - pd.Timedelta(days=self.spec["horizon_days"])).date(),
                    "horizon_days": self.spec["horizon_days"],
                    "forecast_units": round(max(float(pred), 0.0), 2),
                    "model_used": scenario,
                }
        return results

    def predict(self, **req) -> dict:
        return self.predict_many([req], tag_index=False)[0]

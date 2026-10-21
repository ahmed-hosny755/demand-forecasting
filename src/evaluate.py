"""Forecast accuracy metrics."""
import numpy as np


def evaluate(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": np.abs(y - p).mean(),
            "RMSE": np.sqrt(((y - p) ** 2).mean()),
            "WAPE_%": np.abs(y - p).sum() / np.abs(y).sum() * 100,
            "Bias_%": (p - y).sum() / y.sum() * 100}          # > 0 means over-forecasting

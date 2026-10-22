"""Single source of truth for paths, forecast design and feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "demand_clean.csv"      # created by notebook 01
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# ── Forecast design ───────────────────────────────────────────
# NOTE: feature names such as `lag_7` and `epidemic_lag_7` assume HORIZON = 7.
HORIZON = 7                      # predict demand 7 days ahead
LAG_DAYS = [7, 14, 28]           # every lag must be >= HORIZON (checked in features.py)
ROLL_WINDOWS = [7, 28]

TEST_START = "2023-10-01"        # everything before is used for training (train + validation)
SEED = 42
N_ESTIMATORS = 2000              # number of trees found by early stopping in notebook 02 (cap reached)

# ── Features ──────────────────────────────────────────────────
CATEGORICAL = ["category", "store_id", "product_id"]
CALENDAR = ["month", "dayofweek", "week"]
PLANNED = ["price", "discount", "promotion"]                       # planned in advance, known at forecast time
HISTORY = ["lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_28"]

FEATURES_UNKNOWN = CATEGORICAL + CALENDAR + PLANNED + HISTORY + ["epidemic_lag_7"]   # epidemic of target day NOT known
FEATURES_KNOWN = FEATURES_UNKNOWN + ["epidemic"]                                     # epidemic of target day known

DEFAULT_PARAMS = {"max_depth": 5, "learning_rate": 0.1, "subsample": 1.0,
                  "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 10}

# ── Experiment tracking (MLflow) ──────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"     # local database: runs + model registry
MLFLOW_ARTIFACT_DIR = ROOT / "mlartifacts"                              # logged models and files
MLFLOW_EXPERIMENT = "demand-forecasting"

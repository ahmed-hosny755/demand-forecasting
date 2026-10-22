# Retail Demand Forecasting — 7-Day-Ahead Forecasts for 100 Store-Product Series

> **Status:** 🚧 in progress — EDA, modeling, the inventory-decision analysis and experiment tracking (MLflow) are complete; sanity checks (ablation study) are next.

Forecasting daily product demand for a 5-store retailer, with a focus on **leakage-safe time-based validation**, honest baselines, error analysis by regime, category and store, and turning forecasts into **stock targets**.

## TL;DR

- **Task:** Forecast `demand` **7 days ahead** for 100 store-product series (5 stores × 20 products, 76,000 daily records, 2022-01-01 → 2024-01-30).
- **Validation:** Strictly time-based — train < 2023-07-01, validation 2023-07-01 → 2023-09-30, and **test 2023-10-01 → 2024-01-30**. The test period contains epidemic episodes (25.4% of test days vs. 0% in validation), making it a regime-shift stress test.
- **Forecasting results:** XGBoost achieves **8.23% WAPE** when the epidemic flag is known in advance and **9.13%** when it is unknown, versus **43.57%** for a naive "same day last week" forecast and **34.00%** for the per-series mean baseline.
- **Inventory impact (indicative):** The forecast-based stock target using XGBoost with the epidemic flag unknown reduces average excess stock by **84.0% versus a Naive-7-based target**. It holds approximately **61% less stock** than the observed inventory while covering demand on more days (90.05% vs. 86.34%).
- **Important caveat:** The model relies primarily on planned inputs such as price, discount, category and store. Demand-history features contribute very little in this dataset. See [Limitations and honest interpretation](#limitations-and-honest-interpretation).

## Business problem

Stockouts can cause lost sales, while overstock ties up capital. Reliable demand forecasts can help retailers estimate how much inventory to hold and make better replenishment decisions.

This project evaluates both forecasting accuracy and the potential inventory implications of using the resulting predictions.

## Dataset

- **Name:** `demand_forecasting`
- **Source:** TODO — add the original dataset download link before publishing the repository.
- **Size:** 5 stores × 20 products × 760 days = 76,000 daily records.
- **Target:** `demand`.
- **Features:** Price, discount, promotion, competitor pricing, weather, seasonality, epidemic status and inventory.

The original and processed data files are not tracked in Git. Place the original CSV at `data/raw/demand_forecasting.csv`.

## Approach

### 1. Exploratory data analysis

Notebook: `notebooks/01_eda.ipynb`

- Data quality and descriptive statistics.
- Demand drivers and time-series structure.
- Stockout-related analysis.
- Visual exploration of trends and relationships.

### 2. Forecast design and feature engineering

Notebook: `notebooks/02_modeling.ipynb`

- **Forecast horizon:** 7 days ahead.
- **Leakage-safe features:** Calendar variables, planned price/discount/promotion, category, store, product, demand lags and rolling statistics shifted to respect the forecast horizon.
- **Excluded features:** `units_sold`, `units_ordered` and variables derived from these outcomes, as well as same-day `inventory_level`, `competitor_pricing` and `weather`, which are not assumed to be known seven days ahead.
- **Feature sets:**
  - **A — Epidemic known:** Includes the target-day epidemic flag as a known scenario input.
  - **B — Epidemic unknown:** Does not use the target-day epidemic flag.
- **Leakage checks:** Unit tests verify the temporal constraints on history-based features.

### 3. Models and evaluation

The evaluated models are:

- Naive-7
- Rolling mean (7d)
- Series mean (train)
- Ridge (epidemic known)
- XGBoost (epidemic unknown)
- XGBoost (epidemic known)

Model tuning uses the validation period. The final models are refitted on the training and validation data, then evaluated on the held-out test period.

### 4. Inventory decisions

Notebook: `notebooks/03_inventory_decisions.ipynb`

The test-set forecasts are converted into inventory targets using a category-level forecast-error buffer. The analysis compares service levels, average stock, excess inventory and unmet demand across the forecasting policies.

## Results

**Test period:** 2023-10-01 → 2024-01-30 (12,200 rows).

| Model | MAE | RMSE | WAPE (%) | Bias (%) | WAPE reduction vs. Naive-7 |
|---|---:|---:|---:|---:|---:|
| **XGBoost (epidemic known)** | **8.12** | **10.01** | **8.23** | 0.40 | 81.1% |
| XGBoost (epidemic unknown) | 9.01 | 11.22 | 9.13 | 1.19 | 79.0% |
| Ridge (epidemic known) | 26.45 | 34.13 | 26.80 | 1.92 | 38.5% |
| Series mean (train) | 33.56 | 41.84 | 34.00 | 4.57 | 22.0% |
| Rolling mean (7d) | 36.22 | 45.76 | 36.70 | -0.35 | 15.8% |
| Naive-7 | 42.99 | 54.89 | 43.57 | -0.44 | 0.0% |

![Forecast vs actual](reports/figures/16_forecast_vs_actual.png)

### Error analysis

- **Regime changes:** On rows where epidemic status differs from seven days earlier, Naive-7 reaches 60.9% WAPE, compared with 39.1% otherwise. XGBoost remains more stable: 10.9% vs. 8.7% for the epidemic-unknown model and 9.1% vs. 8.0% for the epidemic-known model.
- **Epidemic days:** XGBoost (epidemic known) reaches 11.1% WAPE on epidemic days versus 7.6% on normal days.
- **By category:** Clothing is the easiest category (7.3–7.4% WAPE), while Furniture and Toys are among the hardest (9.2–11.6%).
- **Bias:** All models are relatively close to unbiased, with absolute bias at or below 1.2%, except for the series mean (+4.57%) and Ridge (+1.92%).

### What the model uses

![Permutation importance](reports/figures/17_permutation_importance.png)

Permutation importance, measured as the increase in MAE when a feature is shuffled, shows that `price` dominates, followed by `category`, `product_id`, `store_id` and `discount`.

**Demand-lag and rolling features have almost zero importance.** This suggests that the model's predictive performance depends primarily on the planned inputs available in this dataset. Feature importance describes predictive reliance, not causality.

## From forecasts to inventory decisions

A forecast is useful when it helps improve a business decision. This project converts predictions into a stock target for each store-product-day:

> `target stock = forecast + buffer`

The buffer is the 95th percentile of the forecast error (`demand - forecast`) estimated separately for each category.

- **Calibration period:** 2023-10-01 → 2023-11-30 (6,100 rows; 30% epidemic days).
- **Evaluation period:** 2023-12-01 → 2024-01-30 (6,100 rows; 21% epidemic days).

This separation prevents the evaluation period from being used to estimate the buffers.

**Metrics:**

- **Service level:** Percentage of days when the target stock covers demand.
- **Excess stock:** Units held above demand.
- **Unmet demand:** Units of demand not covered by the target.

### Policy comparison

Evaluation period, with a nominal 95% target level.

| Model used for the target | Service level (%) | Avg stock | Avg excess | Excess (% of demand) | Unmet (% of demand) | Excess reduction vs. Naive-7 |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost (epidemic unknown)** | 90.05 | 115.03 | **13.94** | 13.70 | **0.63** | **84.0%** |
| XGBoost (epidemic known) | 93.90 | 116.00 | 14.63 | 14.38 | 0.35 | 83.2% |
| Ridge (epidemic known) | 93.74 | 155.52 | 54.94 | 54.01 | 1.14 | 36.8% |
| Series mean (train) | 92.30 | 165.65 | 65.72 | 64.60 | 1.77 | 24.4% |
| Rolling mean (7d) | 93.67 | 177.66 | 77.51 | 76.20 | 1.55 | 10.8% |
| Naive-7 | 93.62 | 186.90 | 86.89 | 85.42 | 1.69 | 0.0% |

![Service level trade-off](reports/figures/18_service_level_tradeoff.png)

The XGBoost-based target requires substantially less excess stock than the Naive-7 policy in this simulation.

### Where the savings come from

- **By epidemic status:** Naive-7 has 109 excess units during epidemic days versus 81 on normal days. XGBoost keeps excess around 13–16 units across both regimes.
- **By category:** XGBoost's excess stock is approximately 10–16 units across categories, compared with 69–94 for Naive-7.

### Comparison with observed inventory

| Policy | Service level (%) | Avg stock | Avg excess | Excess (% of demand) | Unmet (% of demand) |
|---|---:|---:|---:|---:|---:|
| Observed inventory | 86.34 | 298.49 | 203.08 | 199.63 | 6.22 |
| Forecast-based target (XGBoost, epidemic unknown, 95%) | 90.05 | 115.03 | 13.94 | 13.70 | 0.63 |

In this indicative comparison, the forecast-based target covers demand on more days (90.05% vs. 86.34%), holds approximately 61% less stock and reduces average excess from 203.08 to 13.94 units per store-product-day.

### Caveats for the inventory results

- **The 95% service-level target was not reached.** The epidemic-unknown model achieved 90.05%. The buffers were calibrated on only two months with a different epidemic mix from the evaluation period. The shortfall is particularly noticeable for Toys (85.2%) and Furniture (87.3%). Longer calibration windows or buffers by category and regime are potential improvements.
- **The observed-inventory comparison is indicative only.** `inventory_level` is the stock recorded on a given day, not necessarily the intended replenishment target. The simulation does not account for lead times, ordering costs, holding costs or stockout costs.
- **The dataset appears synthetic.** The estimated inventory savings should not be interpreted as guaranteed real-world business outcomes.

## Experiment tracking (MLflow)

XGBoost experiments are logged with MLflow so model configurations, evaluation metrics and artifacts can be inspected and compared.

- **Experiment:** `demand-forecasting`.
- **Runs:** The supplied comparison screenshot shows four runs, with the names `epidemic_known` and `epidemic_unknown` appearing twice. The screenshot does not establish the total number of runs in the experiment.
- **Logged parameters:** Feature-set configuration, XGBoost hyperparameters and experiment metadata. The visible comparison includes `min_child_weight`, `n_estimators`, `n_features`, `n_series`, `reg_lambda`, `rows_test`, `rows_train` and `seed`.
- **Feature sets:** Epidemic known and epidemic unknown.
- **Evaluation metrics:** Validation MAE during tuning and final test-set metrics, including MAE, RMSE, WAPE and Bias.
- **Artifacts:** Trained model files in JSON format and the feature specification, where logged by the run.

![MLflow runs comparison](reports/figures/mlflow_runs.png)

The final test-set results are **9.13% WAPE for XGBoost (epidemic unknown)** and **8.23% WAPE for XGBoost (epidemic known)**.

The screenshot confirms that runs can be compared by parameters, but it does not show the validation MAE values or identify the best validation run. Those details should be added only after checking the corresponding MLflow metrics.

> MLflow supports reproducibility and experiment comparison. The test set is reserved for final evaluation rather than model selection.

## Sanity checks

> TODO: Complete the ablation study in section 14 of `02_modeling.ipynb`. Compare test-set WAPE for:
>
> - All features.
> - Without price.
> - Without price, discount and promotion.
> - Without lag features.
> - History features only.
>
> Add the results table and a short interpretation of how much the model relies on planned inputs versus demand history. Remove this section if the experiment is skipped.

## Limitations and honest interpretation

- The data appears synthetic, with complete and highly regular records. Absolute accuracy may not represent production performance.
- Predictive performance mainly reflects how planned inputs such as price, discount, category and store explain demand in this dataset. History-based features contribute very little.
- The evaluation uses a single held-out test period, while the validation period contains no epidemic episodes.
- `demand` is used as defined in the dataset. `units_sold` exceeds `demand` in 27.6% of rows, so `demand` is not assumed to be exact true customer demand.
- Feature importance and observed relationships are associations, not causal estimates.
- Inventory evaluation is a simplified single-period simulation without lead times or costs, and the nominal 95% service-level target was not fully achieved. See [Caveats for the inventory results](#caveats-for-the-inventory-results).

## Project structure

```text
demand-forecasting/
├── data/
│   ├── raw/                  # Original CSV (not tracked)
│   └── processed/            # Cleaned data and predictions (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb          # Data quality, drivers and time-series analysis
│   ├── 02_modeling.ipynb     # Baselines, features, validation and XGBoost
│   └── 03_inventory_decisions.ipynb
├── src/                      # Reusable features, training and evaluation code
├── tests/                    # Unit tests and train/serve consistency checks
├── models/                   # Trained models and feature specification
├── reports/
│   └── figures/              # Forecast, importance, inventory and MLflow figures
├── mlruns/                   # Local MLflow tracking data (not tracked in Git)
├── requirements.txt
└── README.md
```

## How to run

Create and activate a virtual environment, install the dependencies and launch Jupyter:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_eda.ipynb` — creates `data/processed/demand_clean.csv`.
2. `notebooks/02_modeling.ipynb` — creates `data/processed/test_predictions.csv`.
3. `notebooks/03_inventory_decisions.ipynb` — writes `reports/inventory_policy_summary.csv` and `reports/inventory_vs_observed.csv`.

### Browse MLflow experiments

From the project root, run:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Select the `demand-forecasting` experiment to inspect runs, compare parameters and metrics, and review available artifacts.

## Tech stack

Python 3.13 · pandas · NumPy · scikit-learn · XGBoost · MLflow · statsmodels · seaborn / matplotlib · Jupyter

## Roadmap

- [x] 01 — EDA and data quality.
- [x] 02 — Baselines, lag features, time-based validation, XGBoost and error analysis.
- [ ] Sanity checks / ablation study (planned inputs versus demand history).
- [x] 03 — Turn forecasts into inventory decisions (stock targets and service levels).
- [ ] Improve service-level calibration with longer calibration windows and buffers by category and regime.
- [x] Experiment tracking (MLflow).
- [ ] A small API / dashboard.

## Author

**Ahmed Hosny** — [GitHub](https://github.com/ahmed-hosny755)

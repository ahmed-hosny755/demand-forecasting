# Retail Demand Forecasting — 7-Day-Ahead Forecasts for 100 Store-Product Series

> **Status:** 🚧 in progress — EDA and modeling are complete; sanity checks and inventory-decision analysis are next.

Forecasting daily product demand for a 5-store retailer, with a focus on **leakage-safe time-based validation**, honest baselines, and error analysis by regime, category and store.

## TL;DR
- **Task:** forecast `demand` **7 days ahead** for 100 store-product series (5 stores × 20 products, 76,000 daily records, 2022-01-01 → 2024-01-30).
- **Validation:** strictly time-based — train < 2023-07-01, validation 2023-07-01 → 2023-09-30, **test 2023-10-01 → 2024-01-30**. The test period contains epidemic episodes (25.4% of test days vs. 0% in validation), which makes it a regime-shift stress test.
- **Result:** XGBoost reaches **8.2% WAPE** when the epidemic flag is known in advance and **9.1%** when it is not, versus **43.6%** for a naive "same day last week" forecast and **34.0%** for the best baseline (per-series mean) — about **76% lower error** than the best baseline.
- **Important caveat:** the model relies almost entirely on planned inputs (price, discount, category, store) and barely uses demand history. See [Limitations](#limitations-and-honest-interpretation).

## Business problem
Stockouts lose sales and overstock ties up capital. Reliable demand forecasts let the retailer decide how much to stock and when to reorder.

## Dataset
- **Name:** `demand_forecasting` — **Source:** `TODO: add the link where you downloaded it`
- 5 stores × 20 products × 760 days. Target: `demand`. Features: price, discount, promotion, competitor price, weather, seasonality label, epidemic flag, inventory.
- Data files are git-ignored. Place the CSV at `data/raw/demand_forecasting.csv`.

## Approach
1. **EDA** (`01_eda.ipynb`): data quality, demand drivers, time-series structure, stockout analysis.
2. **Forecast design** (`02_modeling.ipynb`):
   - **Horizon:** 7 days. Every history feature is at least 7 days old.
   - **Legal features only:** calendar, planned price/discount/promotion, category/store/product, demand lags (7/14/28 days), rolling mean/std of demand shifted by 7 days, and the epidemic status 7 days earlier.
   - **Excluded on purpose:** `units_sold`, `units_ordered` and anything derived from them (outcomes), plus same-day `inventory_level`, `competitor_pricing` and `weather` (not known 7 days ahead).
   - **Leakage unit test:** assertions verify that the lag and rolling features only use data that is ≥ 7 days old.
   - **Two feature sets:** **A** includes the epidemic flag of the target day (a known scenario input); **B** does not.
3. **Models:** Naive-7, rolling mean, per-series mean, Ridge, and XGBoost tuned with a 15-trial random search on the validation period (early stopping on MAE). The final model is refit on train + validation and evaluated **once** on the test set.

## Results (test set: 2023-10-01 → 2024-01-30, 12,200 rows)

| Model | MAE | RMSE | WAPE % | Bias % | WAPE reduction vs. Naive-7 |
|---|---:|---:|---:|---:|---:|
| **XGBoost (epidemic known)** | **8.12** | **10.01** | **8.23** | 0.40 | 81.1% |
| XGBoost (epidemic unknown) | 9.01 | 11.22 | 9.13 | 1.19 | 79.0% |
| Ridge (epidemic known) | 26.45 | 34.13 | 26.80 | 1.92 | 38.5% |
| Series mean (train) | 33.56 | 41.84 | 34.00 | 4.57 | 22.0% |
| Rolling mean (7d) | 36.22 | 45.76 | 36.70 | −0.35 | 15.8% |
| Naive-7 | 42.99 | 54.89 | 43.57 | −0.44 | 0.0% |

![Forecast vs actual](reports/figures/16_forecast_vs_actual.png)

### Error analysis
- **Regime changes** (23.0% of test rows, where the epidemic status differs from 7 days earlier) break history-based forecasts: Naive-7 has 60.9% WAPE on these rows vs. 39.1% otherwise. XGBoost stays stable: 10.9% vs. 8.7% (epidemic unknown) and 9.1% vs. 8.0% (epidemic known).
- **Epidemic days are harder** than normal days: 11.1% vs. 7.6% WAPE for XGBoost (epidemic known).
- **By category:** Clothing is the easiest (7.3–7.4% WAPE); Furniture and Toys are the hardest (9.2–11.6%).
- **Bias:** all models are close to unbiased (|Bias| ≤ 1.2%), except the per-series mean (+4.6%) and Ridge (+1.9%).

### What the model uses
![Permutation importance](reports/figures/17_permutation_importance.png)

Permutation importance (increase in MAE when a feature is shuffled) shows that `price` dominates, followed by `category`, `product_id`, `store_id` and `discount`. **All demand-lag and rolling features have almost zero importance.** Importance shows what the model *uses*, not what *causes* demand.

## Sanity checks
> TODO: fill in after running section 14 of `02_modeling.ipynb` (ablation study) — one table with the WAPE of "all features", "without price", "without price/discount/promotion", "without lag features" and "only history", plus one sentence on what it shows. Delete this section if you skip it.

## Limitations and honest interpretation
- The data looks **synthetic** (complete, no missing values, regular structure), so absolute accuracy should not be read as production performance.
- The strong result mainly reflects how well the **planned inputs (price, discount, category, store)** explain demand in this dataset. History-based features add almost nothing, so this is closer to *demand modeling from planned inputs* than to classical time-series forecasting.
- Single hold-out period; the validation period contains no epidemic episodes.
- `demand` is used exactly as defined in the dataset; `units_sold` exceeds `demand` in 27.6% of rows, so `demand` is not interpreted as exact "true customer demand".
- Feature importance and effects are **associations**, not causal estimates.

## Project structure
```
demand-forecasting/
├── data/
│   ├── raw/                  # original CSV (not tracked)
│   └── processed/            # cleaned data and predictions (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb          # data quality, drivers, time-series analysis, stockouts
│   ├── 02_modeling.ipynb     # baselines, lag features, time-based validation, XGBoost
│   └── 03_inventory_decisions.ipynb   # forecasts → stock targets and service levels
├── src/                      # reusable code: features, training, evaluation
├── tests/                    # unit tests (data leakage, train/serve consistency)
├── models/                   # trained XGBoost models (JSON) + feature spec
├── reports/                  # figures, metrics and summaries
├── requirements.txt
└── README.md
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Run `notebooks/01_eda.ipynb` first (it creates `data/processed/demand_clean.csv`), then `notebooks/02_modeling.ipynb`.

## Tech stack
Python 3.13 · pandas · NumPy · scikit-learn · XGBoost · statsmodels · seaborn / matplotlib · Jupyter

## Roadmap
- [x] 01 — EDA and data quality
- [x] 02 — Baselines, lag features, time-based validation, XGBoost, error analysis
- [ ] Sanity checks / ablation study (is the signal in price or in history?)
- [ ] 03 — Turn forecasts into inventory decisions (safety stock and service level)
- [ ] Experiment tracking (MLflow) and a small API / dashboard

## Author
**Ahmed Hosny** — [GitHub](https://github.com/ahmed-hosny755)
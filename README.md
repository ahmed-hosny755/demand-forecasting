## Key findings from EDA

**Data**
- 76,000 daily records for **100 store-product series** (5 stores × 20 products), 2022-01-01 → 2024-01-30 (760 days). The panel is complete: no missing values, duplicates or missing days.
- `product_id` alone is ambiguous (it appears under several categories across stores), so each `(store_id, product_id)` pair is treated as one series.
- `units_sold` never exceeds stock, but it exceeds `demand` in 27.6% of rows, so `demand` is used as the target exactly as the dataset defines it.

**What drives demand**
- **Epidemic periods are the largest effect:** mean demand 70.2 vs. 112.9 (**−38%**) on 20% of days, arranged in separate episodes (regime shifts).
- **Category** sets the scale: Groceries 121.0 … Furniture 73.6.
- **Promotion** is associated with **+30%** demand (95.0 → 123.3), consistently across categories (+29% to +31%). Discount acts as a step: ≥ 15% matches the promotion level, with no extra lift beyond it.
- Weather: Sunny 115.2 vs. Snowy 94.0. Store differences are small (100.6–106.9).

**Time-series behaviour**
- No weekly cycle: weekday means differ by under 2%; the ACF decays slowly with no spikes at lags 7/14/21, and the PACF is significant only at lags 1–2.
- The apparent "trend" is mostly the epidemic regime switching on and off.
- Category × month seasonality is partly confounded with epidemic episodes, so it is checked again excluding epidemic days.

**Stockouts**
- **11.4%** of store-product-days end in a stockout, and **17.5%** of total demand (≈ 1.39M units) is unmet.
- Stockouts happen on high-demand days (mean demand 140.8 vs. 99.6).
- Worst store-category pairs: S005 × Clothing (33.0%) and S004 × Electronics (29.6%).

**Implications for modeling:** use a time-based hold-out, compute lags per series and respect the forecast horizon, treat the epidemic regime explicitly, and never use `units_sold`-derived columns as predictors.
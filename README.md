[README.md](https://github.com/user-attachments/files/30876479/README.md)
# Sales Forecast Arena

A project analyzing `simulated_financial_forecasting_data.csv` to test whether macroeconomic indicators can forecast `target_sales` — and an interactive Streamlit dashboard for exploring the result.

**Headline finding:** `sales` correlates with `target_sales` at ~0.98. The five macro/market features (`market_indicator_1`, `market_indicator_2`, `gdp_growth`, `unemployment_rate`, `inflation_rate`) each correlate at under 0.03. A model trained on all features scores R² ≈ 0.96; the same model trained on macro indicators alone scores R² ≈ -0.11 — worse than just guessing the average. This project surfaces that finding rather than hiding it behind a shiny accuracy number.

## Contents

| File | Description |
|---|---|
| `sales_forecasting_signal_analysis.ipynb` | Main analysis notebook: EDA, leakage/redundancy check, two modelling tracks (all-features vs. macro-only), evaluation, feature importance, and a written conclusion |
| `financial_sales_forecasting.ipynb` | Earlier, simpler exploratory notebook (baseline + model comparison, less rigorous framing) |
| `app.py` | Gamified Streamlit dashboard: visualizations, live model training, and an interactive prediction tool |
| `requirements.txt` | Python dependencies for the Streamlit app |
| `simulated_financial_forecasting_data.csv` | Dataset (not included here — supply your own copy, see below) |

## Dataset

1,000 rows, 7 numeric columns, no missing values, no date/time column (this is a cross-sectional dataset, not a time series):

- `sales` — current sales figure
- `market_indicator_1`, `market_indicator_2` — market indicators
- `gdp_growth`, `unemployment_rate`, `inflation_rate` — macroeconomic indicators
- `target_sales` — the value being predicted

## Getting started

### 1. Notebook

Upload `sales_forecasting_signal_analysis.ipynb` to [Google Colab](https://colab.research.google.com/) and run top to bottom. It will prompt you to upload `simulated_financial_forecasting_data.csv` via a file picker.

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels` — all preinstalled in Colab.

### 2. Streamlit dashboard

Place `app.py`, `requirements.txt`, and `simulated_financial_forecasting_data.csv` in the same folder, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What the dashboard does

- **Overview** — data snapshot and summary statistics
- **Visualizations** — correlation heatmap, distribution explorer, sales-vs-target scatter with trendline, macro indicator radar chart, R² confidence gauge, feature importance chart, predicted-vs-actual plot, residual plot, sales trend line
- **Model & Prediction** — adjustable sliders for every feature and a **Predict Sales** button that trains a Random Forest and returns a live prediction, plotted against the historical distribution
- **Insights** — plain-language explanation of the `sales`/`target_sales` collinearity finding
- **Sidebar toggle** — switch the model between *all features* and *macro indicators only* to see the accuracy gap directly (R² drops from ~0.96 to ~-0.11 on this dataset)

## Key takeaway

This dataset does not currently support a genuine macro-driven forecasting model — the apparent accuracy comes almost entirely from the near-duplicate relationship between `sales` and `target_sales`. To make this a real forecasting exercise, the dataset would need a time index (for a true forward-looking target and chronological split), segment identifiers (region/product), and actionable levers like pricing or promotional spend.

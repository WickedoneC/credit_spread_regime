# Corporate Spread Regime Analysis (IG OAS + Treasuries)

This project analyzes U.S. investment-grade (IG) corporate credit conditions using the ICE BofA US Corporate Index option-adjusted spread (OAS) and U.S. Treasury rate/curve features. The core goal is to **identify and predict credit stress regimes** using economically interpretable signals and time-series-appropriate evaluation.

## Financial Motivation

Credit spreads compensate investors for default risk, downgrade risk, liquidity risk, and risk premia. In practice, spread dynamics are closely linked to macro conditions and rate/curve regimes:

- **IG OAS** reflects broad corporate credit risk sentiment and liquidity conditions.
- **10Y Treasury yield** proxies the risk-free rate level and “flight-to-quality” dynamics.
- **10Y–2Y slope** is a classic macro cycle indicator tied to recession risk and tightening/loosening financial conditions.

This project builds a framework to:
1) define historically “stressed” credit regimes via spread thresholds, and  
2) evaluate whether macro rate/curve features provide early warning for future stress.

## Data Sources

All series are pulled from FRED and cached locally (not committed to Git).

- IG Corporate OAS: `BAMLC0A0CM`
- 10Y Treasury Yield: `DGS10`
- 10Y–2Y Curve Slope: `T10Y2Y`

## Repository Structure

```text
credit_spread_regime/
├── data/                # local cached CSVs (gitignored)
├── notebooks/           # EDA and modeling
├── src/                 # reusable code (data ingestion)
│   ├── data_loader.py
│   └── utils.py
├── environment.yml      # conda environment (reproducible)
├── .gitignore
└── README.md
```

## Recreating the Project 

1) Create environment
conda env create -n spread_regime_2501 -f environment.yml
conda activate spread_regime_2501


2) Download/cache data (requires FRED API key)
Create a .env file in the project root:
FRED_API_KEY=YOUR_KEY_HERE

Then run:
python -m src.data_loader

3) Run analysis
Open and run:
notebooks/01_eda_spread_regimes.ipynb


## Modeling Approach
**Regime labeling:**

Stress (level): IG OAS above a historical percentile threshold (e.g., 90th percentile).

Stress-entry (forward): predicts whether stress occurs within the next H trading days, conditional on not being stressed today.

This shift from stress-level classification to stress-entry prediction is important because:
stress levels are persistent and can inflate apparent performance
entry prediction is closer to a real early-warning use case and is materially harder

**Evaluation:**

Walk-forward evaluation by calendar year (no random shuffling).

Metrics: ROC AUC reported by year plus aggregate summaries.

**Current Status:**

Implemented:

* FRED data ingestion and caching

* EDA of spreads vs rates and curve slope

* Baseline models using macro-only, lagged features (leakage-safe)

Walk-forward evaluation for:

* stress-level classification

* stress-entry prediction

Next improvements (in-progress):

* Redefine stress-entry using a minimum-duration stress rule to reduce noise from threshold “churn” and improve economic realism.

* Expand feature set and evaluate non-linear models (e.g., XGBoost, PyTorch).

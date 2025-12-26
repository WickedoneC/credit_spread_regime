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

## Targets: Forward spread widening events

We define binary targets based on **maximum forward widening within a window**.

For a horizon `H` (trading days) and threshold `X` (bps), label at time `t` is:


max(OAS_{t+1..t+H}) − OAS_t ≥ X bps

$$ \max_{\tau \in \{t+1, \dots, t+H\}} (\text{OAS}_{\tau} - \text{OAS}_{t}) \ge X $$


The label is positive if the maximum forward increase in IG OAS over the next H trading days exceeds X basis points relative to today’s level.

Interpretation example (`widen_25bps_21d`):

> “As of today, what is the probability that IG OAS widens by **≥25 bps at some point** over the next **21 trading days**?”

### Locked signals (used going forward)

- `widen_10bps_21d` — primary short-horizon risk overlay
- `widen_25bps_21d` — secondary “stress alert” risk signal
- `widen_50bps_21d` — optional tail “red alert” signal (rare)

Target construction lives in `src/targets.py`, with locked definitions in `src/config.py`.

---


## Methodology

- **Leakage control:** features are lagged (only information available at time `t` is used to predict forward events).
- **Model:** logistic regression baseline (scaled inputs; class-weighted).
- **Validation:** **walk-forward by year** (train on prior years, test on the next year).
- **Metrics:**
  - **Pooled ROC AUC:** computed across all out-of-sample predictions
  - **Weighted AUC:** year-level AUCs weighted by observations

---

## Results: Horizon × threshold grid (pooled ROC AUC)

Pooled AUC results for predicting forward IG OAS widening:

| Threshold (bps) \ Horizon (days) | 21 | 63 | 126 |
|---|---:|---:|---:|
| **10** | **0.6648** | 0.5517 | 0.5050 |
| **25** | **0.6290** | 0.5447 | 0.4249 |
| **50** | **0.6340** | 0.4895 | 0.5225 |

**Key takeaway:** macro variables are most informative for **short-horizon (≈1 month)** spread widening risk. Predictive power decays at longer horizons, consistent with credit outcomes becoming dominated by fundamentals and idiosyncratic factors.
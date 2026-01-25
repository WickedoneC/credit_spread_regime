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

## Updated Modeling, Evaluation, and Next Steps

### What we predict (targets)

We define **binary forward “widening event” targets** based on the *maximum* IG OAS widening within a forward window.

For horizon \(H\) (trading days) and threshold \(X\) (bps), the label at time \(t\) is:

$$
y_t = \mathbb{1}\left[\max_{\tau \in \{t+1,\dots,t+H\}}\left(\mathrm{OAS}_\tau - \mathrm{OAS}_t\right) \ge X\right]
$$

Interpretation example (`widen_25bps_21d`):

> “As of today, what is the probability that IG OAS widens by **≥ 25 bps at some point** over the next **21 trading days**?”

**Why “max widening” instead of end-to-end change?**  
Because in credit, many decisions are driven by **peak-to-trough stress** within a risk window (mark-to-market risk, drawdown control, VaR shocks), not only the terminal level at \(t+H\).

---

### Features (baseline + regime conditioning)

Baseline features are simple and interpretable credit/rates state variables (e.g., OAS level, curve slope, rates level), all constructed to avoid look-ahead.

We also add **regime-conditioned** (leakage-safe) features designed to capture macro-credit state:

- **Credit valuation regime:** rolling percentile and z-score of IG OAS (tight / mid / wide)
- **Curve regime:** inversion indicator (10Y–2Y < 0)
- **Rates volatility regime:** realized vol of daily changes in 10Y yield + high-vol flag
- **Interactions:** inverted×high-vol, wide×inverted, wide×high-vol

All newly created regime features are **lagged by 1 day** to ensure the model only uses information available at decision time.

---

### Evaluation (walk-forward validation)

We use a **walk-forward (expanding window)** evaluation:

- Train on years up to \(t\)
- Evaluate on the next year
- Repeat across the sample
- Report:
  - **Pooled AUC** (all out-of-sample predictions concatenated)
  - **Weighted AUC** (yearly AUC weighted by number of observations)

This mirrors real deployment where models are refit through time and scored out-of-sample.

---

### Current results snapshot (H = 21 days)

Below is a snapshot comparing baseline vs regime-conditioned features for selected targets:

| target | pos_rate | baseline pooled AUC | baseline weighted AUC | regime pooled AUC | regime weighted AUC |
|---|---:|---:|---:|---:|---:|
| widen_10bps_21d | 0.233 | 0.663 | 0.598 | 0.663 | 0.535 |
| widen_25bps_21d | 0.054 | 0.630 | 0.546 | 0.667 | 0.621 |
| widen_50bps_21d | 0.018 | 0.634 | 0.788 | 0.758 | 0.843 |

**Key takeaway:** Regime conditioning improves discrimination for **moderate-to-tail widening events** (25–50 bps), consistent with the idea that macro state variables matter most during stress transitions.

---

### Next steps 

We will focus on **linear models** for the next phase.

1) **Stability plots**
   - Year-by-year AUC (baseline vs regime vs non-linear model)
   - Rolling-window AUC to assess time stability (e.g., 2-year rolling)

2) **Threshold calibration**
   - Precision/recall at operational alert rates (e.g., top 5% risk days)
   - Precision@N (e.g., top 10 / 20 risk days per year), especially for rare 50 bps events

3) **Decision overlays**
   - Translate predicted probabilities into portfolio actions (risk throttle / hedge triggers)
   - Compare a small set of overlay rules and measure outcome differences

4) **Economic backtest (proxy)**
   - Use OAS move capture and/or duration-adjusted spread-move proxy PnL
   - Evaluate whether alerts/overlays reduce adverse widening exposure in practice


## Financial rationale (PM-facing)

### What this project is doing in market terms

This project builds a **forward-looking early-warning indicator** for **IG credit spread risk**, using publicly available market data.

At a high level, we estimate:

> “Given today’s credit/rates backdrop, what is the probability that IG spreads experience a *meaningful widening episode* within the next \(H\) trading days?”

We do this using a binary event target that triggers if **the maximum forward widening** within the horizon exceeds a threshold:

$$
y_t = \mathbf{1}\left[\max_{\tau \in \{t+1,\dots,t+H\}}\left(\mathrm{OAS}_\tau - \mathrm{OAS}_t\right) \ge X\right]
$$

**Why ‘max widening’ matters:** in real portfolios, the risk you manage is often the *in-window drawdown / mark-to-market shock*, not only the end-of-window level. A transient spread blowout can still force risk reduction, hedging, or stop-outs.

---

### Why these inputs are economically meaningful

We start with a small set of variables that map to standard credit and macro intuition:

- **IG OAS level (risk premium / valuation):**  
  When spreads are already wide, the market is pricing in elevated risk; when spreads are tight, credit is more vulnerable to repricing from shocks.
- **Rates level (10Y yield) and curve slope (10Y–2Y):**  
  These proxy the macro regime: growth/inflation expectations, policy tightness, recession risk, and funding conditions. Credit repricing is often regime-dependent (e.g., late-cycle / inversion dynamics).
- **Regime conditioning (state variables rather than raw levels):**  
  We convert levels into *interpretable regimes* such as “OAS is historically wide” or “curve is inverted.” This mirrors how PMs think: **context matters** for interpreting a given spread move.

In other words, we are not trying to “predict spread ticks.” We are trying to classify **risk states** where the market is statistically more prone to produce **large forward widening moves**.

---

### How a PM would use it

The output is a **probability score** each day for each defined event (e.g., widen_25bps_21d). In a production setting, this is typically used as a **risk overlay**:

- **Risk throttle:** reduce credit beta when the probability breaches a calibrated threshold
- **Hedge trigger:** add/scale hedges when the score indicates elevated tail widening risk
- **Position sizing / carry filter:** keep carry risk on only when the signal indicates low near-term widening risk

The goal is not perfect forecasting; it’s **improving decision timing** and **avoiding concentrated exposure into widening regimes**.

---

### Why the evaluation approach matches reality

We evaluate with **walk-forward (out-of-sample) testing** so that each year’s performance is generated using only prior history. This approximates how a strategy would be trained and monitored in a live setting (periodic refits, ongoing scoring).

We report performance both as:
- **Pooled AUC:** all out-of-sample predictions combined
- **Weighted AUC:** average of yearly AUCs weighted by the number of observations

This prevents “one crisis year” from dominating conclusions and forces the signal to prove itself across multiple market environments.


## Run daily (local)

From repo root:

```bash
python -m src.run_daily
```

## Interpreting the daily output

The daily CSV contains one row per locked target:

- `proba`: model-estimated probability that IG OAS will widen by at least X bps over the next 21 trading days.
- `threshold`: calibrated per-year threshold chosen so that, historically, it fires ~Top-N alerts per year.
- `alert`: 1 if `proba >= threshold`.
- `risk_level` mapping:
  - `HIGH`: primary (25 bps) alert triggered
  - `ELEVATED`: secondary (50 bps) alert triggered (but primary not)
  - `LOW`: neither triggered
- `de_risk_exposure`: suggested exposure reduction (e.g., 25%) when risk is HIGH/ELEVATED.


## Method summary

We model forward IG spread widening using a walk-forward-by-year logistic regression. Features are strictly lagged (T-1) to avoid leakage. Thresholds are calibrated per calendar year using a Top-N rule to control alert frequency and make signals operationally usable. Daily output is a compact risk table + decision overlay charts to support discretionary portfolio decisions.


## Current Market Environment (Context)

**As of the most recent trading date**, the model evaluates U.S. Investment Grade credit risk in the context of both recent spread dynamics and the prevailing rates regime.

Over the **last 21 trading days**, IG option-adjusted spreads (OAS):

- **Current level:** 78 bps  
- **21-day change:** -4 bps  
- **Direction:** TIGHTENING

This short-horizon spread movement is provided strictly as **context** for the model’s forward-looking risk signals.

### Relationship to Model Outputs

The daily signals estimate the **probability that IG OAS will widen by at least a specified threshold within the next 21 trading days**, conditional on the current macro–rates and volatility regime:

- **25 bps / 21d (Primary signal)**  
  Captures elevated risk of a meaningful but non-stress widening episode.

- **50 bps / 21d (Secondary signal)**  
  Tail-risk indicator designed to flag potential stress or regime-shift scenarios.

Probabilities are converted into alerts using a **top-N-per-year calibration framework**, ensuring that signal frequency remains stable across market cycles and volatility environments.

### Interpretation Notes

- Recent spread tightening or widening does **not** mechanically imply lower or higher forward risk.
- The model is explicitly designed to identify **risk asymmetries** that may emerge even when recent spread behavior appears benign.
- Signals should be interpreted as **probabilistic risk indicators**, not point forecasts of spread levels or timing.

---

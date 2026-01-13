from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Stress + episode labeling helpers (unchanged)
# =============================================================================

def compute_stress_flag(
    df: pd.DataFrame,
    oas_col: str = "ig_oas",
    stress_q: float = 0.90,
) -> tuple[pd.Series, float]:
    """
    Binary stress flag based on OAS percentile threshold.
    Returns (stress_flag, threshold_value).
    """
    threshold = df[oas_col].quantile(stress_q)
    stress = (df[oas_col] >= threshold).astype(int)
    stress.name = "stress"
    return stress, float(threshold)


def stress_episode_entry(
    stress: pd.Series,
    min_duration: int = 10,
) -> pd.Series:
    """
    Identify entry days for stress episodes that last at least min_duration days.

    Returns 0/1 Series 'stress_entry' where 1 indicates the FIRST day of each qualifying episode.
    """
    s = stress.astype(int)

    # Identify contiguous runs where s changes
    run_id = (s != s.shift(1)).cumsum()

    # Length of each run
    run_len = s.groupby(run_id).transform("size")

    # Qualifying stress days: stress==1 and run length >= min_duration
    qualifying = (s == 1) & (run_len >= min_duration)

    # Entry is the first day of each qualifying run
    entry = (qualifying & (~qualifying.shift(1).fillna(False))).astype(int)
    entry.name = "stress_entry"
    return entry


def entry_within_horizon(
    entry: pd.Series,
    stress: pd.Series,
    horizon: int = 21,
) -> pd.Series:
    """
    Forward-looking label:
      1 if NOT stressed today AND a qualifying episode entry occurs within next 'horizon' days.

    Looks forward starting tomorrow (t+1..t+horizon).
    """
    entry_future = entry.shift(-1).rolling(horizon, min_periods=horizon).max()
    y = ((stress == 0) & (entry_future == 1)).astype(int)
    y.name = f"entry_within_{horizon}d"
    return y


# =============================================================================
# (A) Base feature engineering (rates-only)
# These are the columns your config expects for BASE_FEATURE_COLS.
# =============================================================================

def add_base_features(
    df: pd.DataFrame,
    dgs10_col: str = "dgs10",
    slope_col: str = "t10y2y",
    vol_lookback: int = 20,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Add leakage-safe base (rates-only) features used by the daily model.

    Produces (at minimum, matching src/config.py BASE_FEATURE_COLS):
      - dgs10_lag1
      - t10y2y_lag1
      - dgs10_chg1_bps_lag1
      - t10y2y_chg1_bps_lag1
      - dgs10_vol20_lag1
      - t10y2y_vol20_lag1
      - rate_level_x_slope   (computed from lagged values)

    Notes:
      - dgs10 and t10y2y assumed to be in percent units.
      - chg1_bps is 1-day difference converted to bps (diff * 100).
      - vol20 is rolling std of chg1_bps.
      - lag applied to avoid any lookahead / leakage.
    """
    out = df.copy()

    # 1) Level lags
    out[f"{dgs10_col}_lag{lag}"] = out[dgs10_col].shift(lag)
    out[f"{slope_col}_lag{lag}"] = out[slope_col].shift(lag)

    # 2) 1-day changes (bps) + lag
    # dgs10 and t10y2y are in percent units; convert daily changes to bps via * 100
    out["dgs10_chg1_bps"] = out[dgs10_col].diff() * 100.0
    out["t10y2y_chg1_bps"] = out[slope_col].diff() * 100.0

    out["dgs10_chg1_bps_lag1"] = out["dgs10_chg1_bps"].shift(lag)
    out["t10y2y_chg1_bps_lag1"] = out["t10y2y_chg1_bps"].shift(lag)

    # 3) Realized vol of changes (20d) + lag (bps units)
    out["dgs10_vol20"] = out["dgs10_chg1_bps"].rolling(vol_lookback, min_periods=vol_lookback).std()
    out["t10y2y_vol20"] = out["t10y2y_chg1_bps"].rolling(vol_lookback, min_periods=vol_lookback).std()

    out["dgs10_vol20_lag1"] = out["dgs10_vol20"].shift(lag)
    out["t10y2y_vol20_lag1"] = out["t10y2y_vol20"].shift(lag)

    # 4) Interaction (use lagged values to keep it leakage-safe)
    out["rate_level_x_slope"] = out[f"{dgs10_col}_lag{lag}"] * out[f"{slope_col}_lag{lag}"]

    return out


# =============================================================================
# (B) Regime features (your existing logic) + public wrapper name expected by run_daily.py
# =============================================================================

def _add_regime_features_impl(
    df: pd.DataFrame,
    oas_col: str = "ig_oas",
    slope_col: str = "t10y2y",
    dgs10_col: str = "dgs10",
    oas_lookback: int = 756,   # ~3y trading days
    vol_lookback: int = 20,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Add leakage-safe regime features:
      - OAS percentile / z-score (rolling)
      - Curve inversion indicator
      - Rates realized vol regime
      - Select interactions

    Assumes:
      - OAS in percent units (e.g., 1.25 == 125 bps)
      - dgs10 in percent yield units
      - slope in percent (10Y-2Y)
    """
    out = df.copy()

    # --- 1) OAS valuation / risk premia regime ---
    out["ig_oas_bps"] = out[oas_col] * 100.0

    def _rolling_pct_rank(x: pd.Series) -> float:
        # rank of last observation in the window
        return x.rank(pct=True).iloc[-1]

    out["oas_pctile"] = (
        out["ig_oas_bps"]
        .rolling(oas_lookback, min_periods=max(252, oas_lookback // 3))
        .apply(_rolling_pct_rank, raw=False)
    )

    oas_mean = out["ig_oas_bps"].rolling(oas_lookback, min_periods=max(252, oas_lookback // 3)).mean()
    oas_std = out["ig_oas_bps"].rolling(oas_lookback, min_periods=max(252, oas_lookback // 3)).std()
    out["oas_z"] = (out["ig_oas_bps"] - oas_mean) / oas_std

    out["oas_tight"] = (out["oas_pctile"] <= 0.30).astype(int)
    out["oas_wide"] = (out["oas_pctile"] >= 0.70).astype(int)
    out["oas_mid"] = ((out["oas_pctile"] > 0.30) & (out["oas_pctile"] < 0.70)).astype(int)

    # --- 2) Rates curve regime ---
    out["curve_inverted"] = (out[slope_col] < 0).astype(int)

    # --- 3) Rates volatility regime (bps units) ---
    out["dgs10_chg1_bps"] = out[dgs10_col].diff() * 100.0
    out["dgs10_vol20"] = out["dgs10_chg1_bps"].rolling(vol_lookback, min_periods=vol_lookback).std()

    vol_med = out["dgs10_vol20"].rolling(oas_lookback, min_periods=max(252, oas_lookback // 3)).median()
    out["rates_vol_high"] = (out["dgs10_vol20"] > vol_med).astype(int)

    # --- Interactions ---
    out["inv_x_vol"] = out["curve_inverted"] * out["rates_vol_high"]
    out["wide_x_inv"] = out["oas_wide"] * out["curve_inverted"]
    out["wide_x_vol"] = out["oas_wide"] * out["rates_vol_high"]

    # --- Leakage safety: lag all new regime columns ---
    new_cols = [
        "ig_oas_bps", "oas_pctile", "oas_z", "oas_tight", "oas_mid", "oas_wide",
        "curve_inverted", "dgs10_chg1_bps", "dgs10_vol20", "rates_vol_high",
        "inv_x_vol", "wide_x_inv", "wide_x_vol",
    ]
    for c in new_cols:
        out[f"{c}_lag{lag}"] = out[c].shift(lag)

    return out


def add_regime_features(
    df: pd.DataFrame,
    oas_col: str = "ig_oas",
    slope_col: str = "t10y2y",
    dgs10_col: str = "dgs10",
    oas_lookback: int = 756,
    vol_lookback: int = 20,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Public API expected by src/run_daily.py.

    This wrapper name matches the import:
        from src.features import add_base_features, add_regime_features
    """
    return _add_regime_features_impl(
        df=df,
        oas_col=oas_col,
        slope_col=slope_col,
        dgs10_col=dgs10_col,
        oas_lookback=oas_lookback,
        vol_lookback=vol_lookback,
        lag=lag,
    )

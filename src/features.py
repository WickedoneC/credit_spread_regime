from __future__ import annotations

import pandas as pd


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

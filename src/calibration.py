from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_top_n_by_year(
    oos_y: pd.Series,
    oos_proba: pd.Series,
    n: int = 10,
) -> pd.DataFrame:
    """
    Calibrate a per-year threshold by selecting the Top-N probability days each year.
    Returns a per-year table with threshold and basic alert-quality stats.

    Notes:
      - Threshold is the Nth-highest probability in that year.
      - Alerts count is <= N if there are fewer than N observations in the year.
      - Precision/recall computed against oos_y within each year.
    """
    dfp = pd.DataFrame({"y": oos_y, "p": oos_proba}).dropna()
    if not isinstance(dfp.index, pd.DatetimeIndex):
        raise TypeError("Expected DateTimeIndex for oos_y/oos_proba index.")
    dfp["year"] = dfp.index.year

    rows = []
    for yr, g in dfp.groupby("year"):
        g = g.sort_values("p", ascending=False)

        k = min(n, len(g))
        if k == 0:
            continue

        thr = float(g["p"].iloc[k - 1])  # Nth-highest
        alerts = (g["p"] >= thr).astype(int)

        tp = int(((alerts == 1) & (g["y"] == 1)).sum())
        pos = int((g["y"] == 1).sum())
        n_alerts = int(alerts.sum())

        precision = tp / n_alerts if n_alerts else np.nan
        recall = tp / pos if pos else np.nan

        rows.append(
            {
                "year": int(yr),
                "threshold": thr,
                "alerts": n_alerts,
                "tp": tp,
                "pos": pos,
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def per_year_threshold_series_top_n(
    oos_proba: pd.Series,
    n: int = 10,
) -> pd.Series:
    """
    Returns a time-aligned threshold Series (indexed like oos_proba),
    where each date uses that year's Top-N calibrated threshold.
    """
    dfp = pd.DataFrame({"p": oos_proba}).dropna()
    if not isinstance(dfp.index, pd.DatetimeIndex):
        raise TypeError("Expected DateTimeIndex for oos_proba index.")
    dfp["year"] = dfp.index.year

    thr_by_year = {}
    for yr, g in dfp.groupby("year"):
        g = g.sort_values("p", ascending=False)
        k = min(n, len(g))
        thr_by_year[int(yr)] = float(g["p"].iloc[k - 1]) if k else np.nan

    thr = dfp["year"].map(thr_by_year).astype(float)
    thr.name = f"thr_top{n}"
    return thr.reindex(oos_proba.index)

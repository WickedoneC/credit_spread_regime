from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.calibration import per_year_threshold_series_top_n


def build_daily_risk_table(
    df: pd.DataFrame,
    results: dict,
    targets: list[str],
    variant: str = "regime",
    top_n: int = 10,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Returns a one-row-per-target table for the most recent available OOS date (or `asof` if provided).
    """
    rows = []

    for target in targets:
        key = (target, variant)
        if key not in results:
            raise KeyError(f"Missing results for {key}. Available keys: {list(results.keys())[:5]}...")

        oos_proba = results[key]["oos_proba"].dropna()
        if oos_proba.empty:
            continue

        # choose date
        if asof is None:
            dt = oos_proba.index.max()
        else:
            asof = pd.Timestamp(asof)
            dt = oos_proba[oos_proba.index <= asof].index.max()

        p = float(oos_proba.loc[dt])

        thr_series = per_year_threshold_series_top_n(oos_proba, n=top_n)
        thr = float(thr_series.loc[dt])

        rows.append(
            {
                "asof_date": dt,
                "target": target,
                "variant": variant,
                "proba": p,
                "threshold": thr,
                "alert": int(p >= thr),
                "ig_oas_bps_today": float(df.loc[dt, "ig_oas"] * 100.0) if "ig_oas" in df.columns and dt in df.index else None,
            }
        )

    out = pd.DataFrame(rows).sort_values(["variant", "target"]).reset_index(drop=True)
    return out


def save_daily_outputs(
    df: pd.DataFrame,
    results: dict,
    targets: list[str],
    variant: str = "regime",
    top_n: int = 10,
    out_dir: str | Path = "outputs",
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = build_daily_risk_table(
        df=df,
        results=results,
        targets=targets,
        variant=variant,
        top_n=top_n,
        asof=asof,
    )

    daily.to_csv(out_dir / "daily_risk_table.csv", index=False)
    return daily

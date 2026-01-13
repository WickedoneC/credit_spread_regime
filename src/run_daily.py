from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.calibration import per_year_threshold_series_top_n
from src.config import (
    CALIBRATION_TOP_N_PER_YEAR,
    DAILY_VARIANT,
    DE_RISK_EXPOSURE,
    FEATURE_COLS_REGIME,
    LOCKED_SIGNALS,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
)
from src.data_loader import load_data
from src.features import add_base_features, add_regime_features
from src.reporting import build_overlay_df, plot_decision_overlay
from src.targets import make_forward_widen_target

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fit_logit_and_score(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.Series:
    """
    Walk-forward by year:
      - For each test year, refit using all prior years
      - Score that year's observations out-of-sample

    IMPORTANT:
      - Returns a Series indexed to the FULL df.index (not the dropna subset),
        so callers can safely .loc[df.index.max()] even if it is NaN.
    """
    df_ = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df_[feature_cols]
    y = df_[target_col]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    years = sorted(df_.index.year.unique())

    # ✅ index to full df.index so .loc[asof] never KeyErrors (may be NaN, which is fine)
    proba = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train_idx = df_.index.year.isin(train_years)
        test_idx = df_.index.year == test_year

        # Need both classes to fit
        if y.loc[train_idx].nunique() < 2:
            continue

        pipe.fit(X.loc[train_idx], y.loc[train_idx])
        proba.loc[X.loc[test_idx].index] = pipe.predict_proba(X.loc[test_idx])[:, 1]

    return proba


def map_risk_level(primary_alert: int, secondary_alert: int) -> str:
    if primary_alert == 1:
        return "HIGH"
    if secondary_alert == 1:
        return "ELEVATED"
    return "LOW"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    today = date.today().isoformat()

    # --------------------------------------------------
    # 1) Load + feature engineering
    # --------------------------------------------------
    df = load_data()
    df = add_base_features(df)
    df = add_regime_features(df)

    missing = [c for c in FEATURE_COLS_REGIME if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # --------------------------------------------------
    # 1.5) Build locked target columns (labels)
    # --------------------------------------------------
    for sig in LOCKED_SIGNALS:
        tcol = f"widen_{sig['bps']}bps_{sig['horizon']}d"
        df[tcol] = make_forward_widen_target(
            df=df,
            oas_col="ig_oas",
            horizon=sig["horizon"],
            widen_bps=sig["bps"],
        )

    # --------------------------------------------------
    # 2) Refit + score locked targets
    # --------------------------------------------------
    results: dict[str, pd.Series] = {}
    for sig in LOCKED_SIGNALS:
        target = f"widen_{sig['bps']}bps_{sig['horizon']}d"
        results[target] = fit_logit_and_score(
            df=df,
            feature_cols=FEATURE_COLS_REGIME,
            target_col=target,
        )

    for t in [PRIMARY_TARGET, SECONDARY_TARGET]:
        if t not in results:
            raise KeyError(f"Target {t} not found in results. Available: {list(results.keys())}")

    # --------------------------------------------------
    # 3) Threshold calibration (Top-N per year)
    # --------------------------------------------------
    alerts: dict[str, pd.Series] = {}
    thresholds: dict[str, pd.Series] = {}

    for target, proba in results.items():
        thr = per_year_threshold_series_top_n(proba, n=CALIBRATION_TOP_N_PER_YEAR)
        thresholds[target] = thr
        alerts[target] = (proba >= thr).astype(int)

    # --------------------------------------------------
    # 4) Daily row (as-of date = last *scored* date)
    # --------------------------------------------------
    latest_idx = df.index.max()

    daily_rows = []
    for target in [PRIMARY_TARGET, SECONDARY_TARGET]:
        p = results[target]
        thr = thresholds[target]
        a = alerts[target]

        # Choose an as-of date that has both proba and threshold available
        scored_mask = p.notna() & thr.notna()
        if not scored_mask.any():
            raise RuntimeError(f"No scored dates available for target={target}. Check feature/label NA patterns.")

        asof = scored_mask[scored_mask].index.max()

        daily_rows.append(
            {
                "date": asof.date().isoformat(),
                "target": target,
                "proba": float(p.loc[asof]),
                "threshold": float(thr.loc[asof]),
                "alert": int(a.loc[asof]),
            }
        )

    daily_df = pd.DataFrame(daily_rows)

    primary_alert = int(daily_df.loc[daily_df["target"] == PRIMARY_TARGET, "alert"].iloc[0])
    secondary_alert = int(daily_df.loc[daily_df["target"] == SECONDARY_TARGET, "alert"].iloc[0])

    daily_df["risk_level"] = map_risk_level(primary_alert, secondary_alert)
    daily_df["de_risk_exposure"] = DE_RISK_EXPOSURE
    daily_df["model_variant"] = DAILY_VARIANT

    out_csv = OUTPUT_DIR / f"daily_signal_{today}.csv"
    daily_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # --------------------------------------------------
    # 5) Decision overlays (PNGs)
    # --------------------------------------------------
    for target in [PRIMARY_TARGET, SECONDARY_TARGET]:
        overlay = build_overlay_df(
            df=df,
            results={(target, DAILY_VARIANT): {"oos_proba": results[target]}},
            target_col=target,
            variant=DAILY_VARIANT,
            oas_col="ig_oas",
        )

        plot_decision_overlay(
            overlay_df=overlay,
            threshold=thresholds[target],
            title=f"Decision overlay ({target}, {DAILY_VARIANT})",
            outpath=OUTPUT_DIR / f"decision_overlay_{target}_{today}.png",
            show=False,
        )


if __name__ == "__main__":
    main()

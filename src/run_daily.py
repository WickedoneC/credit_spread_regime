from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    LOCKED_SIGNALS,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
    FEATURE_COLS_REGIME,
    CALIBRATION_TOP_N_PER_YEAR,
    DE_RISK_EXPOSURE,
    DAILY_VARIANT,
)

from src.data_loader import load_data
from src.features import add_base_features, add_regime_features
from src.calibration import per_year_threshold_series_top_n
from src.reporting import (
    build_overlay_df,
    plot_decision_overlay,
)

from pathlib import Path

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
    Refit logistic regression up to T-1 and score all dates OOS-style.
    """
    df_ = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df_[feature_cols]
    y = df_[target_col]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    years = sorted(df_.index.year.unique())
    proba = pd.Series(index=df_.index, dtype=float)

    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train_idx = df_.index.year.isin(train_years)
        test_idx = df_.index.year == test_year

        if y.loc[train_idx].nunique() < 2:
            continue

        pipe.fit(X.loc[train_idx], y.loc[train_idx])
        proba.loc[test_idx] = pipe.predict_proba(X.loc[test_idx])[:, 1]

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
    # 2) Refit + score locked targets
    # --------------------------------------------------
    results = {}

    for sig in LOCKED_SIGNALS:
        target = f"widen_{sig['bps']}bps_{sig['horizon']}d"

        proba = fit_logit_and_score(
            df,
            FEATURE_COLS_REGIME,
            target,
        )

        results[target] = proba

    # --------------------------------------------------
    # 3) Threshold calibration (Top-N per year)
    # --------------------------------------------------
    alerts = {}

    for target, proba in results.items():
        thr = per_year_threshold_series_top_n(
            proba,
            n=CALIBRATION_TOP_N_PER_YEAR,
        )
        alerts[target] = (proba >= thr).astype(int)

    # --------------------------------------------------
    # 4) Daily row (T only)
    # --------------------------------------------------
    latest_idx = df.index.max()

    daily_rows = []

    for target in [PRIMARY_TARGET, SECONDARY_TARGET]:
        daily_rows.append({
            "date": latest_idx.date().isoformat(),
            "target": target,
            "proba": float(results[target].loc[latest_idx]),
            "threshold": float(
                per_year_threshold_series_top_n(
                    results[target],
                    n=CALIBRATION_TOP_N_PER_YEAR,
                ).loc[latest_idx]
            ),
            "alert": int(alerts[target].loc[latest_idx]),
        })

    daily_df = pd.DataFrame(daily_rows)

    primary_alert = int(
        daily_df.loc[daily_df["target"] == PRIMARY_TARGET, "alert"].iloc[0]
    )
    secondary_alert = int(
        daily_df.loc[daily_df["target"] == SECONDARY_TARGET, "alert"].iloc[0]
    )

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
            df,
            proba=results[target],
            alert=alerts[target],
        )

        plot_decision_overlay(
            overlay,
            title=f"Decision overlay ({target}, {DAILY_VARIANT})",
            out_path=OUTPUT_DIR / f"decision_overlay_{target}_{today}.png",
        )


if __name__ == "__main__":
    main()

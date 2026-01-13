from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

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
                "ig_oas_bps_today": float(df.loc[dt, "ig_oas"] * 100.0)
                if "ig_oas" in df.columns and dt in df.index
                else None,
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


def plot_decision_overlay(
    overlay_df: pd.DataFrame,
    threshold: float | pd.Series,
    title: str,
    outpath: str | Path | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
):
    """
    Plot a decision overlay:
      - IG OAS (bps)
      - model probability
      - threshold (scalar or Series aligned to overlay_df.index)
      - alert markers where proba >= threshold

    Parameters
    ----------
    overlay_df : DataFrame
        Must include columns: ['ig_oas_bps', 'oos_proba'] (and optionally 'oos_y')
    threshold : float or pd.Series
        If Series, should be index-aligned to overlay_df.index (will be reindexed).
    title : str
    outpath / out_path : str|Path|None
        Either keyword is accepted. If provided, saves PNG to that path.
    show : bool
        If True, displays plot; otherwise closes figure (recommended for scripts).

    Returns
    -------
    DataFrame
        overlay_df with 'threshold' and 'alert' columns added.
    """
    d = overlay_df.copy()

    if not isinstance(threshold, pd.Series):
        d["threshold"] = float(threshold)
    else:
        d["threshold"] = threshold.reindex(d.index).astype(float)

    d["alert"] = (d["oos_proba"] >= d["threshold"]).astype(int)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(d.index, d["ig_oas_bps"], label="IG OAS (bps)")
    ax1.set_ylabel("IG OAS (bps)")

    ax2 = ax1.twinx()
    ax2.plot(d.index, d["oos_proba"], label="Model proba", linestyle="--")
    ax2.plot(d.index, d["threshold"], label="Threshold", linestyle=":")
    ax2.set_ylabel("Probability")

    # mark alerts
    alerts = d[d["alert"] == 1]
    if not alerts.empty:
        ax2.scatter(alerts.index, alerts["oos_proba"], marker="o")

    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()

    # Accept either outpath or out_path
    save_path = outpath if outpath is not None else out_path
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return d


def walk_forward_proba(
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    min_train_years: int = 5,
) -> dict:
    """
    Walk-forward by year.
    Fits on years[:i], tests on year[i], iterating forward.
    Returns:
      - by_year AUC table
      - pooled/weighted AUCs
      - oos_y: concatenated out-of-sample labels (indexed by date)
      - oos_proba: concatenated out-of-sample predicted probabilities (indexed by date)

    Leakage safety: caller must ensure feature_cols are already lagged appropriately.
    """
    df_ = eval_df.dropna(subset=feature_cols + [target_col]).copy()
    X = df_[feature_cols]
    y = df_[target_col].astype(int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    years = sorted(df_.index.year.unique())
    results = []
    oos_y_parts = []
    oos_proba_parts = []

    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train_idx = df_.index.year.isin(train_years)
        test_idx = df_.index.year == test_year

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # Need both classes in train and test
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        results.append((test_year, auc, int(test_idx.sum()), int(y_test.sum())))

        oos_y_parts.append(y_test)
        oos_proba_parts.append(pd.Series(y_proba, index=y_test.index, name="oos_proba"))

    by_year = pd.DataFrame(results, columns=["year", "auc", "n_obs", "n_pos"])

    weighted_auc = np.average(by_year["auc"], weights=by_year["n_obs"]) if len(by_year) else np.nan
    pooled_auc = roc_auc_score(pd.concat(oos_y_parts), pd.concat(oos_proba_parts)) if oos_y_parts else np.nan

    oos_y = pd.concat(oos_y_parts).sort_index() if oos_y_parts else pd.Series(dtype=int, name="oos_y")
    oos_proba = (
        pd.concat(oos_proba_parts).sort_index() if oos_proba_parts else pd.Series(dtype=float, name="oos_proba")
    )

    oos_y.name = "oos_y"
    oos_proba.name = "oos_proba"

    return {
        "target": target_col,
        "n": int(y.shape[0]),
        "pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else np.nan,
        "weighted_auc": float(weighted_auc),
        "pooled_auc": float(pooled_auc),
        "by_year": by_year,
        "oos_y": oos_y,
        "oos_proba": oos_proba,
    }


def run_locked_targets_walkforward(
    df: pd.DataFrame,
    targets: list[str],
    feature_cols_baseline: list[str],
    feature_cols_regime: list[str],
    min_train_years: int = 5,
) -> dict:
    """
    Convenience runner that returns a results dict keyed by (target, variant).
    Each entry contains oos_y / oos_proba / AUC summaries.
    """
    results: dict[tuple[str, str], dict] = {}

    for tcol in targets:
        out_base = walk_forward_proba(df, feature_cols_baseline, tcol, min_train_years=min_train_years)
        out_reg = walk_forward_proba(df, feature_cols_regime, tcol, min_train_years=min_train_years)

        results[(tcol, "baseline")] = out_base
        results[(tcol, "regime")] = out_reg

    return results


def build_overlay_df(
    df: pd.DataFrame,
    results: dict,
    target_col: str,
    variant: str = "regime",
    oas_col: str = "ig_oas",
) -> pd.DataFrame:
    """
    Builds a single time-indexed df with:
      - ig_oas_bps
      - oos_proba
      - oos_y (optional)
    Used for decision overlays and backtest plumbing.
    """
    base = df.copy()
    base["ig_oas_bps"] = base[oas_col] * 100.0

    res = results[(target_col, variant)]
    oos_proba = res["oos_proba"]

    # oos_y is optional
    if "oos_y" in res:
        oos_y = res["oos_y"]
    elif target_col in df.columns:
        oos_y = df[target_col]
    else:
        oos_y = pd.Series(index=df.index, dtype=float)

    overlay = pd.DataFrame(index=base.index)
    overlay["ig_oas_bps"] = base["ig_oas_bps"]
    overlay["oos_proba"] = pd.Series(oos_proba).reindex(overlay.index)
    overlay["oos_y"] = pd.Series(oos_y).reindex(overlay.index)

    return overlay.dropna(subset=["oos_proba"])

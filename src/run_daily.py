from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from src.data_loader import load_merged_df  # adjust to your actual loader
from src.features import add_regime_features
from src.targets import make_forward_widen_target  # adjust name/signature to your actual target builder
from src.config import LOCKED_SIGNALS
from src.reporting import save_daily_outputs
from src.walkforward import walk_forward_auc  # adjust to your actual module/function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--variant", default="regime", choices=["baseline", "regime"])
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (optional)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base data
    df = load_merged_df()  # must return DateTimeIndex and columns incl ig_oas, dgs10, t10y2y

    # 2) Add regime features if needed
    if args.variant == "regime":
        df = add_regime_features(df, oas_col="ig_oas", slope_col="t10y2y", dgs10_col="dgs10")

    # 3) Ensure targets exist
    targets = []
    for s in LOCKED_SIGNALS:
        col = f"widen_{s['bps']}bps_{s['horizon']}d"
        if col not in df.columns:
            df[col] = make_forward_widen_target(df, oas_col="ig_oas", H=s["horizon"], bps=s["bps"])
        targets.append(col)

    # 4) Run walk-forward per target (linear model only; your existing code)
    results = {}
    for target in targets:
        out = walk_forward_auc(
            df=df,
            feature_cols=None,  # set to your baseline/regime feature list
            target_col=target,
            min_train_years=5,
        )
        results[(target, args.variant)] = out

    # 5) Produce daily outputs
    asof = pd.Timestamp(args.asof) if args.asof else None
    daily = save_daily_outputs(df, results, targets, variant=args.variant, top_n=args.topn, out_dir=out_dir, asof=asof)

    print(daily.to_string(index=False))


if __name__ == "__main__":
    main()

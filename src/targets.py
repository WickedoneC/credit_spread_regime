from __future__ import annotations
import pandas as pd


def make_forward_widen_target(
    df: pd.DataFrame,
    oas_col: str,
    horizon: int,
    widen_bps: int,
) -> pd.Series:
    """
    Binary label: 1 if max OAS widening over next `horizon` days >= `widen_bps`.

    Assumes OAS is stored in percent units (e.g., 1.25 == 125 bps) and converts to bps.
    Label definition:
        max_{t+1..t+horizon}(OAS_bps_future) - OAS_bps_today >= widen_bps
    """
    oas_bps = df[oas_col] * 100.0
    fwd_max = oas_bps.shift(-1).rolling(horizon, min_periods=horizon).max()
    fwd_widen = fwd_max - oas_bps

    y = (fwd_widen >= widen_bps).astype(int)
    y.name = f"widen_{widen_bps}bps_{horizon}d"
    return y
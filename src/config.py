# src/config.py

LOCKED_SIGNALS = [
    {"bps": 25, "horizon": 21},
    {"bps": 50, "horizon": 21},
]

PRIMARY_TARGET = "widen_25bps_21d"
SECONDARY_TARGET = "widen_50bps_21d"

CALIBRATION_TOP_N_PER_YEAR = 10
DE_RISK_EXPOSURE = 0.25

DAILY_VARIANT = "regime"   # daily output is based on regime-conditioned feature set


# --- Locked feature sets (used by notebook + run_daily.py) ---

BASE_FEATURE_COLS = [
    "dgs10_lag1",
    "t10y2y_lag1",
    "dgs10_chg1_bps_lag1",
    "t10y2y_chg1_bps_lag1",   # ✅ FIXED
    "dgs10_vol20_lag1",
    "t10y2y_vol20_lag1",
    "rate_level_x_slope",
]


REGIME_FEATURE_COLS = [
    "oas_pctile_lag1",
    "oas_z_lag1",
    "oas_tight_lag1",
    "oas_mid_lag1",
    "oas_wide_lag1",
    "curve_inverted_lag1",
    "dgs10_vol20_lag1",
    "rates_vol_high_lag1",
    "inv_x_vol_lag1",
    "wide_x_inv_lag1",
    "wide_x_vol_lag1",
]

FEATURE_COLS_REGIME = BASE_FEATURE_COLS + REGIME_FEATURE_COLS

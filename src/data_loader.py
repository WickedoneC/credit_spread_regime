from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def get_fred_client() -> Fred:
    """
    Initialize a Fred client using the API key from .env.
    """
    # Load .env if it exists
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Add it to .env in the project root."
        )

    return Fred(api_key=api_key)


def fetch_series(series_id: str) -> pd.Series:
    """
    Fetch a FRED time series as a pandas Series with a DatetimeIndex.
    """
    fred = get_fred_client()
    s = fred.get_series(series_id)
    s = s.to_frame("value")
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s["value"]


def fetch_and_cache_series(series_map: Dict[str, str]) -> Dict[str, pd.Series]:
    """
    Fetch multiple series and cache them as CSVs under data/.

    series_map: mapping from series_id -> filename (e.g. "BAMLC0A0CM" -> "ig_oas.csv")
    """
    DATA_DIR.mkdir(exist_ok=True)
    results: Dict[str, pd.Series] = {}

    for series_id, filename in series_map.items():
        out_path = DATA_DIR / filename
        if out_path.exists():
            # Load from cache
            df = pd.read_csv(out_path, parse_dates=["date"], index_col="date")
            s = df["value"]
        else:
            # Download from FRED and save
            s = fetch_series(series_id)
            s.to_frame("value").to_csv(out_path)

        results[series_id] = s

    return results



def load_data() -> pd.DataFrame:
    """
    Load and return the full daily modeling DataFrame.

    This is a thin wrapper around existing loader logic so that
    notebooks and scripts (e.g. run_daily.py) share a stable interface.
    """
    return main()


def main() -> pd.DataFrame:
    """
    Load core daily macro series and return a merged DataFrame.
    """
    series_map = {
        "BAMLCOA0CM": "ig_oas.csv",   # IG corporate OAS
        "DGS10": "dgs10.csv",         # 10Y Treasury yield
        "T10Y2Y": "t10y2y.csv",       # 10Y-2Y slope
    }

    data = fetch_and_cache_series(series_map)

    for sid, s in data.items():
        print(f"{sid}: {s.index.min().date()} -> {s.index.max().date()}, {s.shape[0]} obs")

    df = pd.concat(
        {
            "ig_oas": data["BAMLCOA0CM"],
            "dgs10": data["DGS10"],
            "t10y2y": data["T10Y2Y"],
        },
        axis=1,
    ).sort_index()

    return df



if __name__ == "__main__":
    main()

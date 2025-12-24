# def prettyTable(df, col):
#     a = df[col].value_counts(dropna = False)
#     b = df[col].value_counts(normalize = True, dropna = False)
#     c = pd.concat([a, b], axis = 1)
#     c.columns = [col + '_count', col + '_ratio']
#     return c


# src/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


def project_root() -> Path:
    """Return project root assuming this file lives in <root>/src/."""
    return Path(__file__).resolve().parents[1]


def prettyTable(
    df: pd.DataFrame,
    col: str,
    dropna: bool = True,
    normalize: bool = True,
    sort: bool = True,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Frequency table with counts and ratios for a dataframe column.
    """
    s = df[col]
    if dropna:
        s = s.dropna()

    counts = s.value_counts(dropna=not dropna)
    ratios = counts / counts.sum() if normalize else None

    out = pd.DataFrame({"count": counts})
    if normalize:
        out["ratio"] = ratios

    if sort:
        out = out.sort_values("count", ascending=ascending)

    return out

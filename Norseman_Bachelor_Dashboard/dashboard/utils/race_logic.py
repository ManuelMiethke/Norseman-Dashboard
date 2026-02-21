from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

GROUP_ALL = "All"
GROUP_TOP10 = "Top 10"
GROUP_BLACK = "Black Shirt"
GROUP_WHITE = "White Shirt"
GROUP_DNF = "DNF"

DEFAULT_YEARS_WITH_37_5_CUTOFF = {2024, 2025}


def parse_time_to_seconds(value: Any) -> float:
    """Convert hh:mm:ss or mm:ss to seconds; invalid values -> NaN."""
    if pd.isna(value) or value is None:
        return np.nan

    s = str(value).strip()
    if s == "":
        return np.nan

    parts = s.split(":")
    try:
        if len(parts) == 2:
            h = 0
            m, sec = parts
        elif len(parts) == 3:
            h, m, sec = parts
        else:
            return np.nan
        return int(h) * 3600 + int(m) * 60 + int(round(float(sec)))
    except Exception:
        return np.nan


def coerce_year(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def apply_year_filter(df: pd.DataFrame, selected_year: Any, year_col: str = "year") -> pd.DataFrame:
    if selected_year in (None, GROUP_ALL) or year_col not in df.columns:
        return df

    year_int = coerce_year(selected_year)
    if year_int is not None:
        return df[pd.to_numeric(df[year_col], errors="coerce") == year_int]

    return df[df[year_col].astype(str) == str(selected_year)]


def apply_group_filter(
    df: pd.DataFrame,
    selected_group: str,
    *,
    finish_col: str = "finish_type",
    rank_col: str = "overall_rank",
    top10_col: str = "Top10_flag",
) -> pd.DataFrame:
    group = (selected_group or GROUP_ALL).strip()
    if group == GROUP_ALL:
        return df

    if group == GROUP_TOP10:
        if top10_col in df.columns:
            return df[df[top10_col].fillna(False)]
        if rank_col in df.columns:
            return df[pd.to_numeric(df[rank_col], errors="coerce") <= 10]
        return df

    if finish_col not in df.columns:
        return df

    finish = df[finish_col].astype(str).str.lower()
    if group == GROUP_BLACK:
        return df[finish.str.contains("black", na=False)]
    if group == GROUP_WHITE:
        return df[finish.str.contains("white", na=False)]
    if group == GROUP_DNF:
        return df[finish.str.contains("dnf", na=False)]
    return df


def year_uses_37_5_cutoff(
    year_value: Any,
    years_with_37_5: Iterable[int] = DEFAULT_YEARS_WITH_37_5_CUTOFF,
) -> bool:
    year_int = coerce_year(year_value)
    return year_int in set(years_with_37_5) if year_int is not None else False


def run_cutoff_column_for_year(year_value: Any) -> str:
    return (
        "run_37_5km_stavsro_cut_off_time"
        if year_uses_37_5_cutoff(year_value)
        else "run_32_5km_langefonn_time"
    )


def run_cutoff_distance_for_year(year_value: Any) -> float:
    return 37.5 if year_uses_37_5_cutoff(year_value) else 32.5


def run_cutoff_seconds_for_row(
    row: pd.Series,
    *,
    year_col: str = "year",
    run_start_col: str = "run_start_time",
) -> float:
    cutoff_col = run_cutoff_column_for_year(row.get(year_col))
    run_start_s = parse_time_to_seconds(row.get(run_start_col))
    cutoff_s = parse_time_to_seconds(row.get(cutoff_col))

    if np.isnan(run_start_s) or np.isnan(cutoff_s):
        return np.nan

    delta_s = cutoff_s - run_start_s
    return delta_s if delta_s >= 0 else np.nan


def apply_run_cutoff(df: pd.DataFrame, output_col: str = "run_time_cutoff_s") -> pd.DataFrame:
    out = df.copy()
    out[output_col] = out.apply(run_cutoff_seconds_for_row, axis=1)
    return out


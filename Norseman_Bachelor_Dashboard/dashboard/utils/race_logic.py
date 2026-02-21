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
GROUP_CRITICAL_40 = "The Critiacl 40 (Rank 140-180)"
GROUP_CRITICAL_40_ALIASES = {
    GROUP_CRITICAL_40,
    "The Critiacl 40",
    "The Critical 40",
    "The Critical 40 (Rank 140-180)",
    "Critical 40",
}

GROUP_ALIASES = {
    GROUP_TOP10: {GROUP_TOP10, "Top10"},
    GROUP_BLACK: {GROUP_BLACK, "Black"},
    GROUP_WHITE: {GROUP_WHITE, "White"},
    GROUP_DNF: {GROUP_DNF},
    GROUP_CRITICAL_40: GROUP_CRITICAL_40_ALIASES,
}

GROUP_COLOR_SCHEMES = {
    "default": {
        GROUP_TOP10: "#00ff00",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#ffffff",
        GROUP_DNF: "#FF4B4B",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
    "timerelations_title_bg": {
        GROUP_TOP10: "#b7ffb7",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#ffffff",
        GROUP_DNF: "#FF4B4B",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
    "accumulated_line": {
        GROUP_TOP10: "#2ECC71",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#ffffff",
        GROUP_DNF: "#FF4B4B",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
    "pacetabelle_table": {
        GROUP_TOP10: "#b7ffb7",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#ffffff",
        GROUP_DNF: "#FF4B4B",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
    "blackshirt_prob": {
        GROUP_TOP10: "#00ff7f",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#ffffff",
        GROUP_DNF: "#FF4B4B",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
    "histograms": {
        GROUP_TOP10: "lightgreen",
        GROUP_BLACK: "black",
        GROUP_WHITE: "white",
        GROUP_DNF: "red",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
    "rank_progression": {
        GROUP_TOP10: "#00C853",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#FFFFFF",
        GROUP_DNF: "#FF0000",
        GROUP_CRITICAL_40: "#2F7D9C",
    },
}

GROUP_TEXT_COLORS = {
    GROUP_TOP10: "#000000",
    GROUP_BLACK: "#ffffff",
    GROUP_WHITE: "#000000",
    GROUP_DNF: "#ffffff",
    GROUP_CRITICAL_40: "#ffffff",
}

NAMED_COLORS = {
    "frontier_pace": "#8A2BE2",
}

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

    if is_critical_40_group(group):
        if rank_col in df.columns:
            rank = pd.to_numeric(df[rank_col], errors="coerce")
            return df[(rank >= 140) & (rank <= 180)]
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


def is_critical_40_group(group_value: Any) -> bool:
    group = (group_value or "").strip()
    return group in GROUP_CRITICAL_40_ALIASES


def canonical_group_name(group_value: Any) -> str:
    group = (group_value or "").strip()
    for canonical, aliases in GROUP_ALIASES.items():
        if group in aliases:
            return canonical
    return group


def get_group_color(group_value: Any, scheme: str = "default", fallback: str = "#9b9b9b") -> str:
    canonical = canonical_group_name(group_value)
    colors = GROUP_COLOR_SCHEMES.get(scheme, GROUP_COLOR_SCHEMES["default"])
    return colors.get(canonical, fallback)


def get_group_text_color(group_value: Any, fallback: str = "#111111") -> str:
    canonical = canonical_group_name(group_value)
    return GROUP_TEXT_COLORS.get(canonical, fallback)


def get_named_color(name: str, fallback: str = "#9b9b9b") -> str:
    return NAMED_COLORS.get(str(name), fallback)


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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.race_logic import (
    GROUP_CRITICAL_40,
    apply_group_filter,
    apply_year_filter,
    get_group_color,
    get_group_text_color,
    is_critical_40_group,
    parse_time_to_seconds,
)

# --------------------------------------------------
# Konfiguration
# --------------------------------------------------
try:
    import data_store
except Exception:
    data_store = None

BG_GRAY = "#7a7a7a"

GROUP_TOP10 = "Top10"
GROUP_BLACK = "Black"
GROUP_WHITE = "White"
GROUP_CRITICAL = "Critical 40"


# --------------------------------------------------
# Helper-Funktionen
# --------------------------------------------------
def time_to_seconds(t) -> float:
    """Konvertiert 'h:mm:ss' oder 'mm:ss' in Sekunden."""
    return parse_time_to_seconds(t)


def seconds_to_hms(sec: float) -> str:
    """Formatiert Sekunden als h:mm:ss (leer, wenn NaN)."""
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return ""
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}"


def pace_value_from_time(time_sec: float, distance_km: float, mode: str) -> float | None:
    """
    Rechnet Gesamtzeit + Distanz in Pace/Speed um:
    - min_per_100m -> Sekunden pro 100m
    - min_per_km   -> Sekunden pro km
    - kmh          -> km/h
    """
    if time_sec is None or np.isnan(time_sec) or distance_km is None:
        return None

    if mode == "min_per_100m":
        dist_100m = distance_km * 10.0
        if dist_100m <= 0:
            return None
        return time_sec / dist_100m

    if mode == "min_per_km":
        if distance_km <= 0:
            return None
        return time_sec / distance_km

    if mode == "kmh":
        hours = time_sec / 3600.0
        if hours <= 0:
            return None
        return distance_km / hours

    return None


def pace_to_str(val: float | None, mode: str) -> str:
    """Formatiert Pace/Speed-Wert als String."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if mode in ("min_per_100m", "min_per_km"):
        sec = int(round(val))
        m = sec // 60
        s = sec % 60
        unit = "min/100m" if mode == "min_per_100m" else "min/km"
        return f"{m:02d}:{s:02d} {unit}"
    if mode == "kmh":
        return f"{val:.1f} km/h"
    return ""


def summarize_group_median(df: pd.DataFrame, time_col: str) -> dict:
    """Median, Std, Min, Max (in Sekunden)."""
    series = df[time_col].dropna()
    if series.empty:
        return dict(median=np.nan, std=np.nan, min=np.nan, max=np.nan)

    return dict(
        median=series.median(),
        std=series.std(ddof=0),
        min=series.min(),
        max=series.max(),
    )


def speed_string_for_metric_median(stats: dict, key: str, distance_km: float | None, mode: str | None) -> str:
    """
    MEDIAN-basiert:
    - Median/Min/Max: Pace/Speed der jeweiligen Zeit
    - St. dev:       Pace/Speed von (Median - Std) und (Median + Std)
    """
    if mode is None or distance_km is None:
        return ""

    med_t = stats["median"]
    std_t = stats["std"]
    val_t = stats[key]

    if key in ("median", "min", "max"):
        return pace_to_str(pace_value_from_time(val_t, distance_km, mode), mode)

    if key == "std":
        if np.isnan(med_t) or np.isnan(std_t):
            return ""
        slower_t = med_t + std_t
        faster_t = max(med_t - std_t, 1)

        slower_pace = pace_to_str(pace_value_from_time(slower_t, distance_km, mode), mode)
        faster_pace = pace_to_str(pace_value_from_time(faster_t, distance_km, mode), mode)

        return f"+ {slower_pace}\n- {faster_pace}"

    return ""


# --------------------------------------------------
# Styling Tabelle (wie vorher)
# --------------------------------------------------
def style_stats_table_html(df: pd.DataFrame) -> str:
    """
    Dark table + farbige Gruppen-Spalten wie in deinem alten Pace-Table:
    - Top10 grün
    - Black schwarz
    - White weiß
    """
    table_styles = [
        dict(
            selector="table",
            props=[
                ("border-collapse", "collapse"),
                ("width", "100%"),
                ("table-layout", "fixed"),
                ("background-color", "#181818"),
            ],
        ),
        dict(
            selector="th",
            props=[
                ("background-color", "#181818"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("padding", "6px 8px"),
                ("border", "1px solid #333333"),
                ("text-align", "center"),
            ],
        ),
        dict(
            selector="td",
            props=[
                ("padding", "6px 8px"),
                ("border", "1px solid #333333"),
                ("text-align", "center"),
                ("white-space", "pre-line"),
            ],
        ),
        dict(
            selector="th, td",
            props=[("width", "12%")],
        ),
    ]

    styler = df.style.hide(axis="index").set_table_styles(table_styles)

    # Standard dark
    styler = styler.set_properties(**{"background-color": "#181818", "color": "white"})

    # Top10 farbig (inkl speed falls vorhanden)
    top10_cols = [c for c in ["Top10", "Top10 speed"] if c in df.columns]
    if top10_cols:
        styler = styler.set_properties(
            subset=top10_cols,
            **{
                "background-color": get_group_color("Top10", scheme="pacetabelle_table"),
                "color": f"{get_group_text_color('Top10')} !important",
            },
        )

    # Black farbig
    black_cols = [c for c in ["Black", "Black speed"] if c in df.columns]
    if black_cols:
        styler = styler.set_properties(
            subset=black_cols,
            **{
                "background-color": get_group_color("Black", scheme="pacetabelle_table"),
                "color": f"{get_group_text_color('Black')} !important",
            },
        )

    # White farbig
    white_cols = [c for c in ["White", "White speed"] if c in df.columns]
    if white_cols:
        styler = styler.set_properties(
            subset=white_cols,
            **{
                "background-color": get_group_color("White", scheme="pacetabelle_table"),
                "color": f"{get_group_text_color('White')} !important",
            },
        )

    critical_cols = [c for c in ["Critical 40", "Critical 40 speed"] if c in df.columns]
    if critical_cols:
        styler = styler.set_properties(
            subset=critical_cols,
            **{
                "background-color": get_group_color("Critical 40", scheme="pacetabelle_table"),
                "color": f"{get_group_text_color('Critical 40')} !important",
            },
        )

    return styler.to_html()


# --------------------------------------------------
# Boxplot (mit Farben)
# --------------------------------------------------
OUTLINE_GRAY = "#2b2b2b"  # dunkelgrau für Rand + Outlier

def _group_style(group_name: str) -> dict:
    if group_name == GROUP_TOP10:
        return dict(fill=get_group_color("Top10"), line=OUTLINE_GRAY, text=get_group_text_color("Top10"))
    if group_name == GROUP_BLACK:
        return dict(fill=get_group_color("Black"), line=OUTLINE_GRAY, text=get_group_text_color("Black"))
    if group_name == GROUP_WHITE:
        return dict(fill=get_group_color("White"), line=OUTLINE_GRAY, text=get_group_text_color("White"))
    if group_name == GROUP_CRITICAL:
        return dict(fill=get_group_color("Critical 40"), line=OUTLINE_GRAY, text=get_group_text_color("Critical 40"))
    return dict(fill="#cccccc", line=OUTLINE_GRAY, text="#111111")


def build_leg_boxplot(df: pd.DataFrame, time_col: str, leg_name: str, selected_group: str = "All") -> go.Figure:
    """
    Boxplot pro Leg, Gruppen: Top10 / Black / White (DNF raus).
    y = Stunden, Hover = h:mm:ss
    """
    plot_df = df[["finish_type", "Top10_flag", "overall_rank", time_col]].copy()
    plot_df = plot_df.dropna(subset=[time_col])

    def group_label(row):
        if bool(row.get("Top10_flag", False)):
            return GROUP_TOP10
        if is_critical_40_group(selected_group):
            rank = pd.to_numeric(pd.Series([row.get("overall_rank")]), errors="coerce").iloc[0]
            if pd.notna(rank) and 140 <= float(rank) <= 180:
                return GROUP_CRITICAL
        ft = row.get("finish_type", "")
        if ft in (GROUP_BLACK, GROUP_WHITE):
            return ft
        return "Other"

    plot_df["group"] = plot_df.apply(group_label, axis=1)
    if is_critical_40_group(selected_group):
        keep_groups = [GROUP_TOP10, GROUP_BLACK, GROUP_CRITICAL]
    else:
        keep_groups = [GROUP_TOP10, GROUP_BLACK, GROUP_WHITE]
    plot_df = plot_df[plot_df["group"].isin(keep_groups)].copy()

    fig = go.Figure()

    if plot_df.empty:
        fig.update_layout(
            title=leg_name,
            paper_bgcolor=BG_GRAY,
            plot_bgcolor=BG_GRAY,
            font=dict(color="black"),
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        fig.add_annotation(
            text="No data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="black", size=14),
        )
        return fig

    plot_df["hours"] = plot_df[time_col] / 3600.0
    plot_df["hms"] = plot_df[time_col].apply(seconds_to_hms)

    order = [GROUP_TOP10, GROUP_BLACK, GROUP_CRITICAL] if is_critical_40_group(selected_group) else [GROUP_TOP10, GROUP_BLACK, GROUP_WHITE]
    for g in order:
        sub = plot_df[plot_df["group"] == g]
        if sub.empty:
            continue

        gs = _group_style(g)

        fig.add_trace(
            go.Box(
                y=sub["hours"],
                name=g,
                boxmean=False,
                boxpoints="outliers",
                fillcolor=gs["fill"],
                line=dict(color=gs["line"], width=2),                 
                marker=dict(color=gs["line"], size=6, opacity=0.9),  
                customdata=sub["hms"],
                hovertemplate=f"<b>{g}</b><br>Time: %{{customdata}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=leg_name,
        paper_bgcolor=BG_GRAY,
        plot_bgcolor=BG_GRAY,
        font=dict(color="black"),
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )

    fig.update_yaxes(
        title=dict(text="Time (h)", font=dict(color="white")),
        gridcolor="rgba(0,0,0,0.18)",
        zerolinecolor="rgba(0,0,0,0.18)",
        tickfont=dict(color="white"),
    )
    fig.update_xaxes(
        title=dict(text="", font=dict(color="white")),
        gridcolor="rgba(0,0,0,0.18)",
        tickfont=dict(color="white"),
    )

    return fig


# --------------------------------------------------
# Stat-Tabelle pro Leg (Top10/Black/White, DNF raus)
# --------------------------------------------------
def build_leg_stats_table(df: pd.DataFrame, time_col: str, leg_cfg: dict, selected_group: str) -> pd.DataFrame:
    """
    Pro Leg:
    - Median, Std dev, Min, Max (Zeit)
    - Pace/Speed-Spalten nur wenn mode != None (T1/T2 haben None)
    - DNF komplett raus
    """
    df_top10 = df[df["Top10_flag"]]
    df_black = df[df["finish_type"] == GROUP_BLACK]
    df_white = df[df["finish_type"] == GROUP_WHITE]
    df_critical = apply_group_filter(
        df,
        GROUP_CRITICAL_40,
        finish_col="finish_type",
        rank_col="overall_rank",
        top10_col="Top10_flag",
    )

    stats_top10 = summarize_group_median(df_top10, time_col)
    stats_black = summarize_group_median(df_black, time_col)
    stats_white = summarize_group_median(df_white, time_col)
    stats_critical = summarize_group_median(df_critical, time_col)

    metrics_order = [
        ("Median", "median"),
        ("St. dev", "std"),
        ("Min", "min"),
        ("Max", "max"),
    ]

    if selected_group == "DNF":
        return pd.DataFrame(
            [{"Metric": "DNF removed"}]
        )

    rows = []
    for label, key in metrics_order:
        row = {"Metric": label}

        dist = leg_cfg.get("dist_km")
        mode = leg_cfg.get("mode")  # None for T1/T2

        def add_group_cols(group_name: str, stats: dict):
            row[group_name] = seconds_to_hms(stats[key])
            if mode is not None and dist is not None:
                row[f"{group_name} speed"] = speed_string_for_metric_median(stats, key, dist, mode)

        show_critical = is_critical_40_group(selected_group)

        if selected_group in ("All", "Top 10") or show_critical:
            add_group_cols("Top10", stats_top10)
        if selected_group in ("All", "Black Shirt") or show_critical:
            add_group_cols("Black", stats_black)
        if selected_group in ("All", "White Shirt"):
            add_group_cols("White", stats_white)
        if show_critical:
            add_group_cols("Critical 40", stats_critical)

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# Daten laden
# --------------------------------------------------
@st.cache_data
def load_wide() -> pd.DataFrame:
    if data_store is None:
        raise RuntimeError("data_store not available. Place data_store.py next to main.py.")

    df = data_store.df_wide()

    time_cols_map = {
        "swim_time": "swim_time_s",
        "t1_time": "t1_time_s",
        "bike_time": "bike_time_s",
        "t2_time": "t2_time_s",
        "run_time": "run_time_s",
        "zombie_hill_time": "zombie_hill_time_s",
        "overall_time": "overall_time_s",
    }

    for src, dst in time_cols_map.items():
        if src in df.columns:
            df[dst] = df[src].apply(time_to_seconds)
        else:
            df[dst] = np.nan

    # Flat marathon section: Run start -> 25 km (before Zombie Hill climb)
    if "run_start_time" in df.columns and "run_25km_zombie_hill_base_time" in df.columns:
        run_start_s = df["run_start_time"].apply(time_to_seconds)
        run_25_s = df["run_25km_zombie_hill_base_time"].apply(time_to_seconds)
        flat_25_s = run_25_s - run_start_s
        flat_25_s = flat_25_s.where(flat_25_s >= 0, np.nan)
        df["run_flat_0_25km_time_s"] = flat_25_s
    else:
        df["run_flat_0_25km_time_s"] = np.nan

    if "overall_rank" in df.columns:
        df["Top10_flag"] = df["overall_rank"].le(10)
    else:
        df["Top10_flag"] = False

    return df


# --------------------------------------------------
# Streamlit-View
# --------------------------------------------------
def render_pace_boxplots_with_tables(selected_year="All", selected_group="All") -> None:
    header_cols = st.columns([0.92, 0.08])
    with header_cols[0]:
        st.markdown("### Pace")
    with header_cols[1]:
        with st.popover("ℹ️"):
            st.markdown("""
     **What is shown in this chart?**  
    Each boxplot shows the distribution of **leg times** (in hours) for athletes grouped by finish category (**Top 10 / Black / White**).  

     **How to interpret this chart**
    - A **lower box / median line** indicates **faster times**.
    - A **taller box** means **higher variability** (more spread in performances).

     **How this can be used**
    - Compare **Top 10 vs. Black / White** to see where time gaps are largest.
    - Identify legs with **high variability**, which may indicate pacing sensitivity or conditions.
    - Use the **stats expander** (Median / Std / Min / Max) for exact summary values.
    """)

    # Load wide dataframe via Bridge 
    df = load_wide()

    # ----------------------------
    # Year filter from header
    # ----------------------------
    df = apply_year_filter(df, selected_year, year_col="year")

    # ----------------------------
    # Legs: Speed/Pace only if mode != None (T1/T2 have None)
    # ----------------------------
    legs_cfg = [
        {"name": "Swim",        "col": "swim_time_s",        "dist_km": 3.8,              "mode": "min_per_100m"},
        {"name": "T1",          "col": "t1_time_s",          "dist_km": None,             "mode": None},
        {"name": "Bike",        "col": "bike_time_s",        "dist_km": 180.0,            "mode": "kmh"},
        {"name": "T2",          "col": "t2_time_s",          "dist_km": None,             "mode": None},
        {"name": "Run",         "col": "run_time_s",         "dist_km": 42.2,             "mode": "min_per_km"},
        {"name": "Flat Marathon Section (to km 25)", "col": "run_flat_0_25km_time_s", "dist_km": 25.0, "mode": "min_per_km"},
        {"name": "Zombie Hill", "col": "zombie_hill_time_s", "dist_km": 7.5,              "mode": "min_per_km"},
        {"name": "Total",       "col": "overall_time_s",     "dist_km": 3.8 + 180 + 42.2, "mode": "kmh"},
    ]

    # Skip legs with no usable data
    legs_cfg = [cfg for cfg in legs_cfg if cfg["col"] in df.columns and df[cfg["col"]].notna().any()]
    if not legs_cfg:
        st.info("No legs with usable timing data found.")
        return

    # ----------------------------
    # Layout: 2 rows (3 + 3) + optional Zombie Hill under T2
    # ----------------------------
    def _find_cfg(name: str):
        for c in legs_cfg:
            if c["name"] == name:
                return c
        return None

    cfg_swim = _find_cfg("Swim")
    cfg_t1 = _find_cfg("T1")
    cfg_bike = _find_cfg("Bike")
    cfg_t2 = _find_cfg("T2")
    cfg_run = _find_cfg("Run")
    cfg_total = _find_cfg("Total")
    cfg_flat_marathon = _find_cfg("Flat Marathon Section (to km 25)")
    cfg_zombie = _find_cfg("Zombie Hill")

    row1_cfgs = [c for c in [cfg_swim, cfg_t1, cfg_bike] if c is not None]
    row2_cfgs = [c for c in [cfg_t2, cfg_run, cfg_total] if c is not None]

    def _render_leg(cfg: dict):
        fig = build_leg_boxplot(df, cfg["col"], cfg["name"], selected_group=selected_group)
        st.plotly_chart(fig, width="content")

        with st.expander("Show stats (Median / Std / Min / Max)", expanded=False):
            stats_df = build_leg_stats_table(df, cfg["col"], cfg, selected_group)
            st.markdown(style_stats_table_html(stats_df), unsafe_allow_html=True)

    # Row 1: Swim / T1 / Bike
    if row1_cfgs:
        cols1 = st.columns(3, gap="medium")
        for i, cfg in enumerate(row1_cfgs):
            with cols1[i]:
                _render_leg(cfg)

    # Row 2: T2 / Run / Total
    cols2 = st.columns(3, gap="medium")
    for i in range(3):
        with cols2[i]:
            if i < len(row2_cfgs):
                _render_leg(row2_cfgs[i])

            # Zombie Hill toggle: appears under T2 (first col in row2)
            if i == 0:
                show_flat_marathon = False
                show_zombie = False

                if cfg_flat_marathon is not None:
                    show_flat_marathon = st.checkbox("Show Flat Marathon Section (to km 25)", value=False)

                # Zombie Hill toggle: appears under flat-marathon toggle
                if cfg_zombie is not None:
                    show_zombie = st.checkbox("Show Zombie Hill", value=False)

                if show_flat_marathon and cfg_flat_marathon is not None and show_zombie and cfg_zombie is not None:
                    cmp_col1, cmp_col2 = st.columns(2, gap="medium")
                    with cmp_col1:
                        _render_leg(cfg_flat_marathon)
                    with cmp_col2:
                        _render_leg(cfg_zombie)
                else:
                    if show_flat_marathon and cfg_flat_marathon is not None:
                        _render_leg(cfg_flat_marathon)
                    if show_zombie and cfg_zombie is not None:
                        _render_leg(cfg_zombie)

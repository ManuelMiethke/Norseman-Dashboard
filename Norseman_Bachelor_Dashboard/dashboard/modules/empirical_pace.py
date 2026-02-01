import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Data Bridge ---
from data_store import df_long as ds_df_long
from data_store import course_profile as ds_course_profile

# ------------------------------------------------------------
# Column names
# ------------------------------------------------------------
COL_YEAR = "year"
COL_BIB = "bib"
COL_FINISH_TYPE = "finish_type"
COL_OVERALL_RANK = "overall_rank"

COL_SPLIT_KEY = "split_key"
COL_SPLIT_NAME = "station_name"
COL_LEG = "leg"

COL_DIST = "split_distance_km"
COL_RACE_DIST = "race_distance_km"
COL_SPEED = "segment_speed_kmh"

# Course profile columns
COL_COURSE_DIST = "distance_km"
COL_ELEV_NORSE = "elev_norseman_m"
COL_ELEV_HAWAII = "elev_hawaii_m"

# Groups
GROUP_TOP10 = "Top 10"
GROUP_BLACK = "Black Shirt"
GROUP_WHITE = "White Shirt"
GROUP_DNF = "DNF"
COL_GROUP = "group"

# ------------------------------------------------------------
# Pacing / distance configuration
# ------------------------------------------------------------
TARGET_TOTAL_SECS = 13 * 3600 + 51 * 60  # 13:51
PREF_OPTIONS = ["p10", "p25", "p50", "p75", "p90"]

# Default labels = training focus
TRAINING_LABELS = {
    "p10": "Very limited training time",
    "p25": "Limited training time",
    "p50": "Balanced training",
    "p75": "More training time",
    "p90": "Significant training focus",
}
# Expert labels = percentiles (kept for Bike/Run only)
EXPERT_LABELS = {
    "p10": "P10",
    "p25": "P25",
    "p50": "P50",
    "p75": "P75",
    "p90": "P90",
}

# Factors scale leg times; >1 means more time, <1 means less time.
PREF_FACTORS = {
    "p10": 1.12,
    "p25": 1.06,
    "p50": 1.00,
    "p75": 0.94,
    "p90": 0.88,
}

SWIM_DIST_M = 3800.0
SWIM_DIST_KM = SWIM_DIST_M / 1000.0
BIKE_DIST_KM = 180.0
RUN_DIST_KM = 42.2

RUN_DECISION_DIST_KM = 37.5
RUN_DECISION_COURSE_KM = SWIM_DIST_KM + BIKE_DIST_KM + RUN_DECISION_DIST_KM

COURSE_MAX_KM = 221.3  

# Years where the Black-Shirt decision point is at 37.5 km
CUTOFF_37_5_YEARS = {2024, 2025}

# Expert swim label heuristic: CSS
CSS_DELTA_SEC_PER_100M = 10  # e.g., 2:05/100m race pace -> CSS ~1:55/100m

# Expert run equivalents (fixed anchors, "fresh flat marathon fitness")
# Matches expectation: P90 ~3:00, P10 ~4:30
RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF = {
    "p90": 2 * 3600 + 45 * 60,  # 2:45  (very strong runner)
    "p75": 3 * 3600 + 10 * 60,  # 3:10
    "p50": 3 * 3600 + 40 * 60,  # 3:40  (ambitious median)
    "p25": 4 * 3600 + 5  * 60,  # 4:05
    "p10": 4 * 3600 + 30 * 60,  # 4:30  (finish-focused runner)
}

# Expert bike equivalents (fixed anchors, "FTP W/kg fitness")
# Based on chosen anchors:
# P90 3.8, P75 3.5, P50 3.1, P25 2.7, P10 2.4
BIKE_FTP_WKG_BY_PREF = {
    "p90": 3.8,
    "p75": 3.5,
    "p50": 3.1,
    "p25": 2.7,
    "p10": 2.4,
}

# Optional: interpretability helper (not used as input)
DEFAULT_BIKE_IF = 0.75  # typical long-distance bike intensity factor

# ------------------------------------------------------------
# Styling constants (local per-figure)
# ------------------------------------------------------------
PANEL_BG = "#7a7a7a"

def apply_plot_bg(fig: go.Figure, bg: str = PANEL_BG) -> go.Figure:
    fig.update_layout(plot_bgcolor=bg, paper_bgcolor=bg)
    return fig

def apply_local_panel_styles(panel_bg: str = PANEL_BG):
    st.markdown(
        f"""
        <style>
        div[data-testid="stDataFrame"] {{
            background: {panel_bg};
            border-radius: 10px;
            padding: 10px;
        }}
        div[data-testid="stExpander"] {{
            background: {panel_bg};
            border-radius: 10px;
            padding: 6px;
        }}
        div[data-testid="stExpander"] > details {{
            background: {panel_bg};
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Empirical baseline Black Shirt leg times (in seconds)
# ------------------------------------------------------------
BASELINE_LEG_SECS = {
    "swim": 70 * 60,                # 1:10:00
    "t1": 5 * 60,                   # 0:05:00
    "bike": 6 * 3600 + 40 * 60,     # 6:40:00
    "t2": 4 * 60,                   # 0:04:00
    "run": 4 * 3600 + 50 * 60,      # 4:50:00
}

# ------------------------------------------------------------
# Helper: group column
# ------------------------------------------------------------
def ensure_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[COL_GROUP] = "Other"
    finish = df[COL_FINISH_TYPE].astype(str)

    df.loc[finish.str.upper().str.contains("DNF"), COL_GROUP] = GROUP_DNF
    mask_finisher = ~finish.str.upper().str.contains("DNF")

    df.loc[
        mask_finisher & finish.str.lower().str.contains("black"),
        COL_GROUP
    ] = GROUP_BLACK

    df.loc[
        mask_finisher & finish.str.lower().str.contains("white"),
        COL_GROUP
    ] = GROUP_WHITE

    df.loc[
        mask_finisher & (pd.to_numeric(df[COL_OVERALL_RANK], errors="coerce") <= 10),
        COL_GROUP
    ] = GROUP_TOP10

    return df

# ------------------------------------------------------------
# Time & pace formatting
# ------------------------------------------------------------
def format_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def format_pace_swim(total_swim_secs: float) -> str:
    sec_per_100m = total_swim_secs / (SWIM_DIST_M / 100.0)
    total = int(round(sec_per_100m))
    m = total // 60
    s = total % 60
    return f"{m:d}:{s:02d} min/100m"

def format_pace_run(total_run_secs: float) -> str:
    sec_per_km = total_run_secs / RUN_DIST_KM
    total = int(round(sec_per_km))
    m = total // 60
    s = total % 60
    return f"{m:d}:{s:02d} min/km"

def calc_bike_speed_kmh(total_bike_secs: float) -> float:
    return BIKE_DIST_KM / (total_bike_secs / 3600.0)

# ------------------------------------------------------------
# Expert labels helpers (Swim CSS + Run marathon equivalent)
# ------------------------------------------------------------
def _fmt_mmss(sec: float) -> str:
    sec = int(round(sec))
    m = sec // 60
    s = sec % 60
    return f"{m:d}:{s:02d}"

def _css_label_from_swim_time(swim_secs: float) -> str:
    # Show ONLY CSS in expert mode radio (derived, heuristic)
    sec_per_100m = swim_secs / (SWIM_DIST_M / 100.0)
    css_sec = max(0.0, sec_per_100m - float(CSS_DELTA_SEC_PER_100M))
    return f"{_fmt_mmss(css_sec)}/100m"

def _format_hm(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h:d}:{m:02d}"

def _run_marathon_label_with_pace(pref: str) -> str:
    pref = _normalize_pref(pref)
    secs = RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF.get(
        pref,
        RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF["p50"]
    )

    # marathon pace
    sec_per_km = secs / MARATHON_DIST_KM
    m = int(sec_per_km // 60)
    s = int(round(sec_per_km % 60))

    return f"{_format_hm(secs)} ({m}:{s:02d}/km)"


def _bike_wkg_label_from_pref(bike_pref: str) -> str:
    pref = _normalize_pref(bike_pref)
    wkg = float(BIKE_FTP_WKG_BY_PREF.get(pref, BIKE_FTP_WKG_BY_PREF["p50"]))
    return f"{wkg:.1f} W/kg"


# ------------------------------------------------------------
# Expert mode: Run equivalents (GAP + implied penalty)
# ------------------------------------------------------------
MARATHON_DIST_KM = 42.195

def _pace_sec_per_km_from_time(total_secs: float, dist_km: float) -> float:
    dist_km = float(dist_km)
    if dist_km <= 0:
        return float("nan")
    return float(total_secs) / dist_km

def _format_pace_mmss_per_km(sec_per_km: float) -> str:
    if not np.isfinite(sec_per_km) or sec_per_km <= 0:
        return "n/a"
    return f"{_fmt_mmss(sec_per_km)}/km"

def _run_gap_from_plan(run_secs: float) -> str:
    # GAP here is a flat-equivalent pace for the planned run time (read-only equivalence)
    sec_per_km = _pace_sec_per_km_from_time(run_secs, RUN_DIST_KM)
    return _format_pace_mmss_per_km(sec_per_km)

def _implied_penalty_from_plan_and_equiv(run_secs: float, run_pref: str) -> float:
    """
    penalty = planned / fresh_equiv - 1
    (read-only interpretability number)
    """
    pref = _normalize_pref(run_pref)
    fresh = float(RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF.get(pref, RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF["p50"]))
    if fresh <= 0:
        return float("nan")
    return float(run_secs) / fresh - 1.0

# ------------------------------------------------------------
# Leg time scaling + preferences
# ------------------------------------------------------------
def scale_legs_to_target(
    baseline_leg_secs: dict,
    target_total_secs: int = TARGET_TOTAL_SECS
) -> dict:
    baseline_total = float(sum(baseline_leg_secs.values()))
    if baseline_total <= 0:
        return baseline_leg_secs.copy()
    scale = target_total_secs / baseline_total
    return {k: v * scale for k, v in baseline_leg_secs.items()}

def _pref_to_factor(pref: str, weak_factor: float, strong_factor: float) -> float:
    pref = str(pref).lower()
    if pref in PREF_FACTORS:
        return PREF_FACTORS[pref]
    if pref.startswith("weak"):
        return PREF_FACTORS["p25"]
    if pref.startswith("strong"):
        return PREF_FACTORS["p75"]
    return PREF_FACTORS["p50"]

def _normalize_pref(pref: str) -> str:
    """
    Map legacy or unknown values to the closest of our five percentiles.
    """
    pref = str(pref).lower().strip()
    if pref in PREF_OPTIONS:
        return pref
    if pref.startswith("weak"):
        return "p25"
    if pref.startswith("strong"):
        return "p75"
    return "p50"

def apply_pref_biases(
    base_leg_secs: dict,
    swim_pref: str,
    bike_pref: str,
    run_pref: str,
) -> dict:
    factors = {
        "swim": _pref_to_factor(swim_pref, PREF_FACTORS["p25"], PREF_FACTORS["p75"]),
        "bike": _pref_to_factor(bike_pref, PREF_FACTORS["p25"], PREF_FACTORS["p75"]),
        "run":  _pref_to_factor(run_pref,  PREF_FACTORS["p25"], PREF_FACTORS["p75"]),
        "t1": 1.0,
        "t2": 1.0,
    }

    pre = {leg: base_leg_secs[leg] * factors.get(leg, 1.0)
           for leg in base_leg_secs.keys()}
    pre_sum = sum(pre.values())
    if pre_sum <= 0:
        return base_leg_secs.copy()

    scale_total = TARGET_TOTAL_SECS / pre_sum
    return {leg: secs * scale_total for leg, secs in pre.items()}

def apply_pref_biases_explore(
    base_leg_secs: dict,
    swim_pref: str,
    bike_pref: str,
    run_pref: str,
) -> dict:
    # Explore mode: no rescale to 13:51 -> total time changes naturally
    factors = {
        "swim": _pref_to_factor(swim_pref, PREF_FACTORS["p25"], PREF_FACTORS["p75"]),
        "bike": _pref_to_factor(bike_pref, PREF_FACTORS["p25"], PREF_FACTORS["p75"]),
        "run":  _pref_to_factor(run_pref,  PREF_FACTORS["p25"], PREF_FACTORS["p75"]),
        "t1": 1.0,
        "t2": 1.0,
    }
    return {leg: base_leg_secs[leg] * factors.get(leg, 1.0) for leg in base_leg_secs.keys()}

# ------------------------------------------------------------
# Build per-split median speeds & scale them to target leg times
# ------------------------------------------------------------
def _effective_cutoff_years(selected_year) -> set[int]:
    """
    If a selected_year is given and it's one of the known 37.5km years,
    use only that year for the empirical median. Otherwise default to {2024, 2025}.
    """
    try:
        y = int(selected_year)
    except Exception:
        return set(CUTOFF_37_5_YEARS)

    return {y} if y in CUTOFF_37_5_YEARS else set(CUTOFF_37_5_YEARS)

def build_blackshirt_segment_pacing(
    long_df: pd.DataFrame,
    scaled_leg_secs: dict,
    selected_year=None,
) -> pd.DataFrame:
    """
    For each split of each leg (swim/bike/run):
    - Take median segment_speed_kmh of all Black Shirt finishers.
      IMPORTANT: only use years where the Black-Shirt decision was at 37.5 km (2024/2025),
      optionally narrowed to selected_year if it is one of those years.
    - Build segments [start_km, end_km] along the course.
    - Scale speeds per leg so that total leg times match scaled_leg_secs.
      (Run only up to 37.5 km).
    """
    df = long_df.copy()

    if COL_YEAR in df.columns:
        df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
        df = df[df[COL_YEAR].isin(_effective_cutoff_years(selected_year))].copy()

    df = df[df[COL_GROUP] == GROUP_BLACK].copy()

    df[COL_LEG] = df[COL_LEG].astype(str)
    df = df[df[COL_LEG].str.lower().isin(["swim", "bike", "run"])]

    for col in [COL_SPEED, COL_RACE_DIST]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[COL_SPEED, COL_RACE_DIST])
    df = df[(df[COL_SPEED] >= 1.0) & (df[COL_SPEED] <= 80.0)]

    grouped = (
        df.groupby([COL_LEG, COL_RACE_DIST], as_index=False)[COL_SPEED]
          .median()
    )
    grouped["leg_lower"] = grouped[COL_LEG].str.lower()

    leg_offsets = {
        "swim": 0.0,
        "bike": SWIM_DIST_KM,
        "run": SWIM_DIST_KM + BIKE_DIST_KM,
    }

    segments = []
    leg_time_secs = {"swim": 0.0, "bike": 0.0, "run": 0.0}

    for leg in ["swim", "bike", "run"]:
        gl = grouped[grouped["leg_lower"] == leg].copy()
        if gl.empty:
            continue

        gl = gl.sort_values(COL_RACE_DIST)
        prev = leg_offsets[leg]

        for row in gl.itertuples():
            cur = float(getattr(row, COL_RACE_DIST))

            if leg == "run":
                if prev >= RUN_DECISION_COURSE_KM:
                    break
                cur = min(cur, RUN_DECISION_COURSE_KM)

            seg_dist = cur - prev
            if seg_dist <= 0:
                prev = cur
                continue

            speed = float(getattr(row, COL_SPEED))
            time_secs = (seg_dist / speed) * 3600.0
            leg_time_secs[leg] += time_secs

            segments.append({
                "leg": leg,
                "start_km": prev,
                "end_km": cur,
                "seg_dist_km": seg_dist,
                "base_speed_kmh": speed,
            })

            prev = cur

    target_secs = {
        "swim": scaled_leg_secs["swim"],
        "bike": scaled_leg_secs["bike"],
        "run": scaled_leg_secs["run"] * (RUN_DECISION_DIST_KM / RUN_DIST_KM),
    }

    scale_factor = {}
    for leg in ["swim", "bike", "run"]:
        base_t = leg_time_secs.get(leg, 0.0)
        tgt_t = target_secs.get(leg, 0.0)
        scale_factor[leg] = (base_t / tgt_t) if (base_t > 0 and tgt_t > 0) else 1.0

    for seg in segments:
        leg = seg["leg"]
        seg["speed_kmh"] = seg["base_speed_kmh"] * scale_factor.get(leg, 1.0)

    return pd.DataFrame(segments)

# ------------------------------------------------------------
# Overview table
# ------------------------------------------------------------
def render_overview_table(scaled_leg_secs: dict, fixed_total: bool = True):
    swim_time = scaled_leg_secs["swim"]
    t1_time = scaled_leg_secs["t1"]
    bike_time = scaled_leg_secs["bike"]
    t2_time = scaled_leg_secs["t2"]
    run_time = scaled_leg_secs["run"]

    total_secs = sum(scaled_leg_secs.values())
    bike_kmh = calc_bike_speed_kmh(bike_time)

    time_at_decision_secs = total_secs

    data = [{
        "Swim Time": format_hms(swim_time),
        "Swim Pace (min/100m)": format_pace_swim(swim_time),
        "T1 Time": format_hms(t1_time),
        "Bike Time": format_hms(bike_time),
        "Bike Pace (km/h)": f"{bike_kmh:.1f} km/h",
        "T2 Time": format_hms(t2_time),
        "Run Time": format_hms(run_time),
        "Run Pace (min/km)": format_pace_run(run_time),
        "Time at 37.5 km run": format_hms(time_at_decision_secs),
    }]
    df = pd.DataFrame(data)

    if fixed_total:
        st.caption(
            f"Leg distribution based on empirical Black Shirt medians "
            f"(years 2024/2025 with 37.5km cut-off), scaled to a total of "
            f"{format_hms(total_secs)} (target 13:51)."
        )
    else:
        st.caption(
            f"Explore mode: leg distribution based on empirical Black Shirt medians "
            f"(years 2024/2025 with 37.5km cut-off). Total time is **unlocked** "
            f"and currently sums to {format_hms(total_secs)}."
        )

    st.markdown(
        """
        <style>
        .emp-pace-table {
            font-size: 28px;
            line-height: 1.9;
            border-collapse: collapse;
            width: 100%;
        }
        .emp-pace-table th,
        .emp-pace-table td {
            font-size: 28px;
            padding: 14px 16px;
            text-align: left;
        }
        .emp-pace-table thead th {
            background: #7a7a7a;
            color: #f4f4f4;
        }
        .emp-pace-table tbody tr:nth-child(odd) {
            background: rgba(255, 255, 255, 0.04);
        }
        .emp-pace-table tbody tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    table_html = df.to_html(index=False, classes="emp-pace-table")
    st.markdown(table_html, unsafe_allow_html=True)

# ------------------------------------------------------------
# Detailed figure: course elevation + recommended speed
# ------------------------------------------------------------
def render_detail_figure(
    long_df: pd.DataFrame,
    course_df: pd.DataFrame,
    scaled_leg_secs: dict,
    selected_year=None,
):
    with st.expander("Show detailed pacing profile (course & speed)", expanded=False):
        seg_df = build_blackshirt_segment_pacing(long_df, scaled_leg_secs, selected_year=selected_year)

        # Speed segments (step-like)
        x_vals = []
        y_vals = []
        for row in seg_df.itertuples():
            x_vals.extend([row.start_km, row.end_km, np.nan])
            y_vals.extend([row.speed_kmh, row.speed_kmh, np.nan])

        course_plot_df = course_df[course_df[COL_COURSE_DIST] <= COURSE_MAX_KM].copy()
        x_e = course_plot_df[COL_COURSE_DIST].to_numpy()
        y_e = course_plot_df[COL_ELEV_NORSE].to_numpy()

        fig = go.Figure()

        # ------------------------------------------------------------
        # Elevation as SHAPE (guaranteed background via layer="below")
        # ------------------------------------------------------------
        if len(x_e) > 1:
            path_parts = [f"M {float(x_e[0])},{float(y_e[0])}"]
            path_parts += [f"L {float(x)},{float(y)}" for x, y in zip(x_e[1:], y_e[1:])]
            path_parts += [f"L {float(x_e[-1])},0", f"L {float(x_e[0])},0", "Z"]
            elev_path = " ".join(path_parts)

            fig.add_shape(
                type="path",
                path=elev_path,
                xref="x",
                yref="y2",
                fillcolor="rgba(31, 111, 178, 0.18)",
                line=dict(color="rgba(31, 111, 178, 0.60)", width=2),
                layer="below",
            )

        # Split-station lines 
        if COL_SPLIT_KEY in long_df.columns and COL_RACE_DIST in long_df.columns:
            split_positions = (
                long_df.groupby(COL_SPLIT_KEY)[COL_RACE_DIST]
                      .mean()
                      .dropna()
                      .unique()
            )
            for x in sorted(pos for pos in split_positions if 0 <= pos <= COURSE_MAX_KM):
                fig.add_vline(
                    x=float(x),
                    line_dash="dot",
                    line_width=1,
                    line_color="rgba(220,220,220,0.4)",
                    layer="below",
                )

        # Black-Shirt decision line (fine above)
        fig.add_vline(
            x=RUN_DECISION_COURSE_KM,
            line_dash="dash",
            line_width=1,
            annotation_text="Black Shirt decision (37.5 km run)",
            annotation_position="top",
        )

        # ------------------------------------------------------------
        # Speed trace (FOREGROUND)
        # ------------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Recommended speed (km/h)",
                line=dict(width=5, color="orange"),
                hovertemplate="Speed: %{y:.1f} km/h<extra></extra>",
                connectgaps=False,
            )
        )

        fig.update_layout(
            xaxis=dict(
                title="Distance (km)",
                range=[0, COURSE_MAX_KM],
                tickmode="array",
                tickvals=[0, 50, 100, 150, 200, COURSE_MAX_KM],
                ticktext=["0", "50", "100", "150", "200", f"{COURSE_MAX_KM:.1f}"],
                gridcolor="rgba(0,0,0,0.15)",
                zerolinecolor="rgba(0,0,0,0.15)",
            ),
            yaxis=dict(
                title="Speed (km/h)",
                gridcolor="rgba(0,0,0,0.15)",
                zerolinecolor="rgba(0,0,0,0.15)",
            ),
            yaxis2=dict(
                title="Elevation (m)",
                overlaying="y",
                side="right",
                gridcolor="rgba(0,0,0,0.0)",
                zerolinecolor="rgba(0,0,0,0.15)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        apply_plot_bg(fig, PANEL_BG)
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)



# ------------------------------------------------------------
# Preferences from session_state
# ------------------------------------------------------------
def _get_current_prefs_from_state():
    swim_pref = _normalize_pref(st.session_state.get("emp_pace_swim_pref", "p50"))
    bike_pref = _normalize_pref(st.session_state.get("emp_pace_bike_pref", "p50"))
    run_pref  = _normalize_pref(st.session_state.get("emp_pace_run_pref",  "p50"))
    return swim_pref, bike_pref, run_pref

# ------------------------------------------------------------
# Main render function 
# ------------------------------------------------------------
def render_13h_empirical_pace(selected_year=None, selected_group=None):
    """
    Bridge-compatible signature:
      empirical_pace.render_13h_empirical_pace(selected_year, selected_group)

    selected_group is currently not used in this module (the blueprint is Black-Shirt-based),
    but we accept it to keep the page interface consistent.
    """
    apply_local_panel_styles(PANEL_BG)

    col_title, col_info = st.columns([0.92, 0.08])
    with col_title:
        st.header("13:51 Black Shirt empirical pacing")
    with col_info:
        with st.popover("ℹ️"):
            st.markdown(
                """
**What is shown here?**

This section shows a **Black Shirt empirical pacing blueprint** for a **13:51 finish time**.

- **Baseline**: median Black Shirt pacing patterns derived **only from years with a 37.5 km Black-Shirt decision rule (2024 & 2025)**.
- **Scaling**: all segment speeds and leg times are **scaled to match the target finish time**.

**What you see below**
- A **summary table** with leg times and paces.
- A **detailed pacing profile** combining recommended speed (orange) and course elevation (blue).

**How to use this**
- Translate historic Black Shirt performance into a realistic personal pacing plan.
- Adjust training focus per discipline to see how pacing shifts while keeping the same total time (13:51).
                """
            )

    # Load from Data Store
    long_df = ds_df_long()
    course_df = ds_course_profile()

    # Ensure group column exists (data_store returns raw CSV)
    if COL_GROUP not in long_df.columns:
        long_df = ensure_group_column(long_df)
    else:
        # still normalize group if it exists but might be outdated
        long_df = ensure_group_column(long_df)

    base_leg_secs = scale_legs_to_target(BASELINE_LEG_SECS, TARGET_TOTAL_SECS)

    swim_pref, bike_pref, run_pref = _get_current_prefs_from_state()

    # Toggles live in the expander, but values are needed for computation
    explore_mode = bool(st.session_state.get("emp_pace_explore_mode", False))
    expert_mode = bool(st.session_state.get("emp_pace_expert_mode", False))

    if explore_mode:
        scaled_leg_secs = apply_pref_biases_explore(base_leg_secs, swim_pref, bike_pref, run_pref)
    else:
        scaled_leg_secs = base_leg_secs
        if (swim_pref, bike_pref, run_pref) != ("p50", "p50", "p50"):
            scaled_leg_secs = apply_pref_biases(base_leg_secs, swim_pref, bike_pref, run_pref)

    # render the overview ONCE 
    render_overview_table(scaled_leg_secs, fixed_total=(not explore_mode))

    with st.expander("Optional: adjust distribution (training focus / expert mode)", expanded=False):
        # Explore toggle first (session_state is the source of truth)
        st.toggle("Explore mode (unlock total time)", key="emp_pace_explore_mode")
        explore_mode = bool(st.session_state.get("emp_pace_explore_mode", False))

        # Expert mode only meaningful/visible in Explore mode
        if explore_mode:
            st.toggle("Expert mode (metrics + expert labels)", key="emp_pace_expert_mode")
        else:
            st.session_state["emp_pace_expert_mode"] = False
        expert_mode = bool(st.session_state.get("emp_pace_expert_mode", False))

        if explore_mode:
            st.markdown("Explore mode: preferences change leg times and the total time (no 13:51 rescaling).")
        else:
            st.markdown(
                "Choose where you spent more or less training time. "
                "This redistributes leg times, but the total stays fixed at 13:51."
            )

        col_swim, col_bike, col_run = st.columns(3)

        # Labels
        label_map = EXPERT_LABELS if expert_mode else TRAINING_LABELS

        # Short training-time labels
        SHORT_TRAINING_LABELS = {
            "p10": "Very limited",
            "p25": "Limited",
            "p50": "Balanced",
            "p75": "More",
            "p90": "Significant",
        }

        # Build expert-mode radio labels (derived/fixed)
        swim_css_labels = {}
        bike_wkg_labels = {}
        run_marathon_labels = {}

        cur_swim_pref = _normalize_pref(st.session_state.get("emp_pace_swim_pref", swim_pref))
        cur_bike_pref = _normalize_pref(st.session_state.get("emp_pace_bike_pref", bike_pref))
        cur_run_pref  = _normalize_pref(st.session_state.get("emp_pace_run_pref",  run_pref))

        if expert_mode:
            # Swim: vary Swim option, keep Bike/Run at current values (derived CSS)
            for opt in PREF_OPTIONS:
                tmp = apply_pref_biases_explore(base_leg_secs, opt, cur_bike_pref, cur_run_pref) if explore_mode \
                    else apply_pref_biases(base_leg_secs, opt, cur_bike_pref, cur_run_pref)
                swim_css_labels[opt] = _css_label_from_swim_time(tmp["swim"])

            # Bike: FIXED FTP W/kg anchors by option
            for opt in PREF_OPTIONS:
                bike_wkg_labels[opt] = _bike_wkg_label_from_pref(opt)

            # Run: FIXED fresh-marathon equivalents by option (anchors)
            for opt in PREF_OPTIONS:
                run_marathon_labels[opt] = _run_marathon_label_with_pace(opt)

        def _radio(col, label, state_key, current_pref):
            if expert_mode and label.lower().startswith("swim"):
                return col.radio(
                    label,
                    PREF_OPTIONS,
                    index=PREF_OPTIONS.index(current_pref),
                    format_func=lambda v: swim_css_labels.get(v, v),
                    key=state_key,
                )
            if expert_mode and label.lower().startswith("bike"):
                return col.radio(
                    label,
                    PREF_OPTIONS,
                    index=PREF_OPTIONS.index(current_pref),
                    format_func=lambda v: bike_wkg_labels.get(v, v),
                    key=state_key,
                )
            if expert_mode and label.lower().startswith("run"):
                return col.radio(
                    label,
                    PREF_OPTIONS,
                    index=PREF_OPTIONS.index(current_pref),
                    format_func=lambda v: run_marathon_labels.get(v, v),
                    key=state_key,
                )
            return col.radio(
                label,
                PREF_OPTIONS,
                index=PREF_OPTIONS.index(current_pref),
                format_func=lambda v: SHORT_TRAINING_LABELS.get(v, v),
                key=state_key,
            )

        # Column titles (cleaner)
        swim_title = "Swim (CSS)" if expert_mode else "Swim Training Time"
        bike_title = "Bike (FTP (W/kg))" if expert_mode else "Bike Training Time"
        run_title  = "Run (Road Marathon Time)" if expert_mode else "Run Training Time"

        swim_pref_sel = _normalize_pref(_radio(col_swim, swim_title, "emp_pace_swim_pref", cur_swim_pref))
        bike_pref_sel = _normalize_pref(_radio(col_bike, bike_title, "emp_pace_bike_pref", cur_bike_pref))
        run_pref_sel  = _normalize_pref(_radio(col_run,  run_title,  "emp_pace_run_pref",  cur_run_pref))

        # Recompute plan immediately based on the selected values + mode toggles
        if explore_mode:
            scaled_leg_secs = apply_pref_biases_explore(base_leg_secs, swim_pref_sel, bike_pref_sel, run_pref_sel)
        else:
            if (swim_pref_sel, bike_pref_sel, run_pref_sel) == ("p50", "p50", "p50"):
                scaled_leg_secs = base_leg_secs
            else:
                scaled_leg_secs = apply_pref_biases(base_leg_secs, swim_pref_sel, bike_pref_sel, run_pref_sel)

        """if expert_mode:
            # keep only the useful expert blocks (Run Expert) below; no "Applied" and no long caption
            pass

        if expert_mode:
            # --- Run expert equivalents (GAP + implied penalty) ---
            gap = _run_gap_from_plan(scaled_leg_secs["run"])
            implied_pen = _implied_penalty_from_plan_and_equiv(scaled_leg_secs["run"], run_pref_sel)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Run (Expert)**")
                st.write(f"Equivalent GAP: **{gap}**")
                st.caption("GAP here is a flat-equivalent pace for interpretation (not an input).")

            with col_b:
                st.markdown("**Fresh marathon equivalent (fixed)**")
                fresh = RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF.get(_normalize_pref(run_pref_sel), RUN_FRESH_MARATHON_EQUIV_SECS_BY_PREF["p50"])
                st.write(f"~ **{format_hms(fresh)}**")
                if np.isfinite(implied_pen):
                    st.caption(f"Implied slowdown vs fresh: **+{int(round(implied_pen*100))}%** (planned / fresh − 1).")
                else:
                    st.caption("Implied slowdown vs fresh: n/a")"""

    render_detail_figure(long_df, course_df, scaled_leg_secs, selected_year=selected_year)

import math

import numpy as np
import pandas as pd
import streamlit as st

# --- Data Bridge ---
from data_store import df_model as ds_df_model

# =========================================
# 0. Constants
# =========================================
SWIM_DIST_KM = 3.8          
BIKE_DIST_KM = 180.0        
RUN_STAVSRO_DIST_KM = 37.5  
TOTAL_STAVSRO_DIST_AFTER_SWIM = BIKE_DIST_KM + RUN_STAVSRO_DIST_KM  # 217.5 km after Swim

# Stavsro cut-off time 
STAVSRO_CUTOFF_TIME_S = 13 * 3600 + 51 * 60  # 13:51:00

# Fixed T2 assumption (only relevant if athlete is still on the bike)
T2_FIXED_TIME_S = 4 * 60  # 4 minutes

# =========================================
# 1. Data prep
# =========================================
@st.cache_data(show_spinner=False)
def load_probs_df_from_store(selected_year: int | None = None) -> pd.DataFrame:
    """
    Load model/probability long DF from the Data Store and optionally filter by year.
    Expected columns (as in your model CSV):
      - p_black (float)
      - y_true (0/1)
      - race_distance_km
      - cum_time_seconds
      - finish_type
      - leg
      - segment_speed_kmh
      - (optional) year
    """
    df = ds_df_model().copy()

    if "p_black" in df.columns:
        df["p_black"] = pd.to_numeric(df["p_black"], errors="coerce")

    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

    if "race_distance_km" in df.columns:
        df["race_distance_km"] = pd.to_numeric(df["race_distance_km"], errors="coerce")

    if "cum_time_seconds" in df.columns:
        df["cum_time_seconds"] = pd.to_numeric(df["cum_time_seconds"], errors="coerce")

    # Optional year filter (only if column exists and selected_year provided)
    if selected_year is not None and "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"] == int(selected_year)].copy()

    df = df.dropna(subset=["p_black", "y_true", "race_distance_km", "cum_time_seconds"], how="any")

    return df


@st.cache_data(show_spinner=False)
def compute_empirical_parameters(df: pd.DataFrame) -> dict:
    """
    Computes median parameters for Black-Shirts:
    - Run pace (min/km)
    - Bike speed (km/h)
    - Target time at Stavsro (seconds since start)
    """
    if df.empty:
        return {
            "run_pace_med_min_per_km": float("nan"),
            "bike_speed_med_kmh": float("nan"),
            "target_stavsro_time_s": float(STAVSRO_CUTOFF_TIME_S),
        }

    mask_black = df["finish_type"].astype(str).str.lower().eq("black")

    # Run median pace
    run_black = df[mask_black & (df["leg"].astype(str).str.lower() == "run")].copy()
    run_black["segment_speed_kmh"] = pd.to_numeric(run_black["segment_speed_kmh"], errors="coerce")
    run_black = run_black.dropna(subset=["segment_speed_kmh"])
    run_black = run_black[run_black["segment_speed_kmh"] > 0]

    if len(run_black) > 0:
        run_black["pace_min_per_km"] = 60.0 / run_black["segment_speed_kmh"]
        run_pace_med = float(run_black["pace_min_per_km"].median())
    else:
        run_pace_med = float("nan")

    # Bike median speed
    bike_black = df[mask_black & (df["leg"].astype(str).str.lower() == "bike")].copy()
    bike_black["segment_speed_kmh"] = pd.to_numeric(bike_black["segment_speed_kmh"], errors="coerce")
    bike_black = bike_black.dropna(subset=["segment_speed_kmh"])
    bike_black = bike_black[bike_black["segment_speed_kmh"] > 0]

    bike_speed_med = float(bike_black["segment_speed_kmh"].median()) if len(bike_black) > 0 else float("nan")

    # Stavsro cut-off time (fixed)
    target_stavsro_time_s = float(STAVSRO_CUTOFF_TIME_S)

    return {
        "run_pace_med_min_per_km": run_pace_med,
        "bike_speed_med_kmh": bike_speed_med,
        "target_stavsro_time_s": target_stavsro_time_s,
    }


# =========================================
# 2. Helper
# =========================================
def parse_time_to_seconds(time_str: str) -> int | None:
    """Expected hh:mm (optional hh:mm:ss)."""
    time_str = (time_str or "").strip()
    if not time_str:
        return None
    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            h, m = map(int, parts)
            s = 0
        elif len(parts) == 3:
            h, m, s = map(int, parts)
        else:
            raise ValueError
    except ValueError:
        raise ValueError("Please use the format hh:mm (or hh:mm:ss).")
    return h * 3600 + m * 60 + s


def format_pace_min_per_km(pace: float | None) -> str:
    if pace is None or not math.isfinite(pace):
        return "–"
    if pace <= 0:
        return "0:00 min/km"
    total_seconds = int(round(pace * 60))
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:d}:{s:02d} min/km"


def format_speed_kmh(speed: float | None) -> str:
    if speed is None or (isinstance(speed, float) and math.isnan(speed)):
        return "–"
    if speed == float("inf"):
        return "∞ km/h"
    if speed <= 0:
        return "–"
    return f"{speed:0.1f} km/h"


def estimate_black_shirt_probability(
    df: pd.DataFrame, race_km: float, current_time_s: float
) -> tuple[float, float, int]:
    """
    Estimate Black Shirt probability via N nearest neighbors in (distance,time) space.
    Returns: (mean_model_prob, empirical_ratio, sample_size)
    """
    if df.empty:
        return float("nan"), float("nan"), 0

    km_scale = 5.0        # 5 km ~ 1 unit
    time_scale = 1800.0   # 30 min ~ 1 unit

    d_km = (df["race_distance_km"] - race_km) / km_scale
    d_t = (df["cum_time_seconds"] - current_time_s) / time_scale
    dist2 = d_km**2 + d_t**2

    N = 80
    nearest_idx = dist2.nsmallest(min(N, len(df))).index
    df_window = df.loc[nearest_idx]

    prob_model = float(df_window["p_black"].mean())
    prob_empirical = float(df_window["y_true"].mean())
    n = int(len(df_window))

    # Fallback if NaNs
    if not math.isfinite(prob_model):
        prob_model = float(df["p_black"].mean())
    if not math.isfinite(prob_empirical):
        prob_empirical = float(df["y_true"].mean())

    return prob_model, prob_empirical, n


def compute_required_paces(
    km_after_swim: float,
    current_time_s: float,
    bike_speed_med_kmh: float,
    run_pace_med_min_per_km: float,
    target_stavsro_time_s: float,
) -> tuple[str, float | None, float | None, bool]:
    """
    km_after_swim = distance since swim exit (bike + run).
    We assume bike and run are faster/slower vs median by a shared factor g.
    Returns:
      leg: "bike", "run", "pre_swim", "beyond"
      bike_speed_req_kmh
      run_pace_req_min_per_km
      infeasible
    """
    remaining_time_s = target_stavsro_time_s - current_time_s

    if km_after_swim < 0:
        return "pre_swim", None, None, True

    if km_after_swim < BIKE_DIST_KM:
        leg = "bike"
        remaining_time_s -= T2_FIXED_TIME_S  
        remaining_bike_km = BIKE_DIST_KM - km_after_swim
        remaining_run_km = RUN_STAVSRO_DIST_KM
    elif km_after_swim <= TOTAL_STAVSRO_DIST_AFTER_SWIM:
        leg = "run"
        remaining_bike_km = 0.0
        run_km_done = km_after_swim - BIKE_DIST_KM
        remaining_run_km = max(0.0, RUN_STAVSRO_DIST_KM - run_km_done)
    else:
        return "beyond", None, None, True

    if remaining_time_s <= 0 or remaining_run_km < 0:
        return leg, None, None, True

    # Median times for remaining distance
    t_bike_med_s = 0.0
    if remaining_bike_km > 0 and math.isfinite(bike_speed_med_kmh) and bike_speed_med_kmh > 0:
        t_bike_med_s = (remaining_bike_km / bike_speed_med_kmh) * 3600.0

    if not (math.isfinite(run_pace_med_min_per_km) and run_pace_med_min_per_km > 0):
        return leg, None, None, True

    t_run_med_s = remaining_run_km * run_pace_med_min_per_km * 60.0
    t_total_med_s = t_bike_med_s + t_run_med_s
    if t_total_med_s <= 0:
        return leg, None, None, True

    g = remaining_time_s / t_total_med_s
    infeasible = g < 0.7  # >30% faster than median

    bike_speed_req = None
    if remaining_bike_km > 0 and math.isfinite(bike_speed_med_kmh) and bike_speed_med_kmh > 0:
        bike_speed_req = bike_speed_med_kmh / g

    run_pace_req = run_pace_med_min_per_km * g
    return leg, bike_speed_req, run_pace_req, infeasible


# =========================================
# 3. Main render function (Bridge signature)
# =========================================
def render_catchup(selected_year=None, selected_group=None):
    """
    Bridge-compatible signature for Page 04:
      catchup.render_catchup(selected_year, selected_group)

    selected_group is not used here (tool is generic), but accepted for consistency.
    """
    import streamlit.components.v1 as components

    # --- Load from Data Store ---
    df_probs = load_probs_df_from_store(selected_year=selected_year)
    params = compute_empirical_parameters(df_probs)

    run_pace_med = params["run_pace_med_min_per_km"]
    bike_speed_med = params["bike_speed_med_kmh"]
    target_stavsro_time_s = params["target_stavsro_time_s"]

    st.markdown(
    """
    <style>

    /* Keep page black */
    [data-testid="stAppViewContainer"]{
        background-color:#000000 !important;
    }

    /* ONLY the catchup container becomes gray */
    .catchup-panel{
        background-color:#7a7a7a !important;
        border-radius:7px !important;
        padding:8px 18px 10px 18px !important;
        margin-top:4px !important;
    }

    .catchup-panel *{
        color:#ffffff !important;
    }

    .catchup-panel input{
        background-color:#1a1a1a !important;
        border:1px solid #ffffff !important;
        color:#ffffff !important;
        border-radius:6px !important;
        padding:0.35rem 0.5rem !important;
    }

    .catchup-panel input::placeholder{
        color:#bbbbbb !important;
    }

    .catchup-panel input:focus{
        border:1px solid #00bfff !important;
        box-shadow:0 0 6px rgba(0,191,255,0.4) !important;
        outline:none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    # ---------------------------------------------------------
    # Catch Up panel (isolated container)
    # ---------------------------------------------------------
    with st.container():
        st.markdown(
    '<div data-catchup-anchor="1" style="height:0;margin:0;padding:0;"></div>',
    unsafe_allow_html=True
)

        components.html(
    """
    <script>
    (function () {
      const MAX_TRIES = 30;      // ~3s total
      const INTERVAL_MS = 100;

      let tries = 0;
      const timer = setInterval(() => {
        tries += 1;

        const doc = window.parent.document;
        const anchors = doc.querySelectorAll('[data-catchup-anchor="1"]');
        const anchor = anchors[anchors.length - 1];
        if (!anchor) {
          if (tries >= MAX_TRIES) clearInterval(timer);
          return;
        }

        // start from the element container hosting the markdown
        const el =
          anchor.closest('div[data-testid="stElementContainer"]') ||
          anchor.closest('div[data-testid="element-container"]') ||
          anchor.parentElement;

        if (!el) {
          if (tries >= MAX_TRIES) clearInterval(timer);
          return;
        }

        // walk up to the nearest suitable container
        let p = el.parentElement;
        while (p && p !== doc.body) {
          const tid = p.getAttribute && p.getAttribute("data-testid");

          // Stop conditions (avoid coloring the whole page)
          if (tid === "stAppViewContainer" || tid === "stMain" || tid === "stApp") {
            clearInterval(timer);
            return;
          }

          // The container we want to color
          if (tid === "stContainer" || tid === "stBlock" || tid === "stVerticalBlock") {
            p.classList.add("catchup-panel");
            clearInterval(timer);
            return;
          }

          p = p.parentElement;
        }

        if (tries >= MAX_TRIES) clearInterval(timer);
      }, INTERVAL_MS);
    })();
    </script>
    """,
    height=0,
)




        # Header row
        col_title, col_info = st.columns([8, 1], vertical_alignment="center")
        with col_title:
            st.markdown(
                "<h2 style='margin-top:0.2rem; margin-bottom:0.25rem;'>What are the odds?</h2>",
                unsafe_allow_html=True,
            )
        with col_info:
            with st.popover("ℹ️"):
                st.markdown(
                    """
**What is shown in this tool?**

Estimate **Black Shirt chances** and the **required pace** to still reach **Stavsro (37.5 km run)**.

- **Inputs**
  - **Bike km (after swim)**: distance already ridden on the bike
  - **Run km**: distance already run (after completing the full 180 km bike)
  - **Race time**: elapsed time since race start

**How the probability works**
We look at historical athletes with **similar distance & time** and show:
- Model-based Black Shirt probability
- Empirical share of Black Shirts in that local neighborhood
                    """
                )

        if df_probs.empty:
            st.error("Model data is empty (after filters). Check the Data Store / selected year filter.")
            return

        st.markdown(
    "<p style='font-size:0.85rem; margin:0.2rem 0;'>"
    "Estimate Black Shirt chances and required pace from the current race status."
    "<em>Only works after the swim (distance after swim ≥ 0 km).</em>"
    "</p>",
    unsafe_allow_html=True,
)


        # Inputs: Bike | Run | Time
        st.markdown("**Current position of your athlete:**")
        col_bike_km, col_run_km, col_time = st.columns([1, 1, 2])

        km_error = None
        time_error = None
        km_after_swim = None
        current_time_s = None

        bike_km_done = None
        run_km_done = None

        with col_bike_km:
            bike_input_str = st.text_input(
                "Bike km",
                value="",
                placeholder="e.g. 50",
                help="0 km = swim exit. 50 km = 50 km on the bike.",
                label_visibility="visible",
            )
            if bike_input_str.strip():
                try:
                    bike_km_done = int(bike_input_str)
                except ValueError:
                    km_error = "Please enter whole numbers (e.g. Bike: 50 or Run: 5)."

        with col_run_km:
            run_input_str = st.text_input(
                "Run km",
                value="",
                placeholder="e.g. 5",
                help="0 km = run start (after the full 180 km bike). 5 km = 5 km on the run.",
                label_visibility="visible",
            )
            if run_input_str.strip():
                try:
                    run_km_done = int(run_input_str)
                except ValueError:
                    km_error = "Please enter whole numbers (e.g. Bike: 50 or Run: 5)."

        with col_time:
            time_input_str = st.text_input(
                "Race time at this point (hh:mm)",
                value="",
                placeholder="hh:mm",
                help="Time since race start.",
            )
            if time_input_str.strip():
                try:
                    current_time_s = parse_time_to_seconds(time_input_str)
                except ValueError as e:
                    time_error = str(e)

        # Resolve distance choice
        if (bike_km_done is not None) and (run_km_done is not None):
            km_error = "Please fill in either Bike km OR Run km (not both)."
        elif bike_km_done is not None:
            km_after_swim = bike_km_done
        elif run_km_done is not None:
            km_after_swim = BIKE_DIST_KM + run_km_done

        if km_error:
            st.error(km_error)
        if time_error:
            st.error(time_error)

        # Compute if valid
        if (km_after_swim is not None) and (current_time_s is not None):
            race_km = SWIM_DIST_KM + km_after_swim

            prob_model, prob_emp, sample_size = estimate_black_shirt_probability(
                df_probs, race_km, current_time_s
            )

            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Black Shirt probability (model)", f"{prob_model * 100:0.1f} %")
            with col_prob2:
                st.metric(
                    "Black Shirt share in similar athletes",
                    f"{prob_emp * 100:0.1f} %",
                    help=f"Based on {sample_size} historical athletes with similar distance & time.",
                )

            st.markdown("---")

            st.markdown(
                "<h4 style='margin-bottom:0.5rem;'>Required pace until Stavsro (37.5 km run)</h4>",
                unsafe_allow_html=True,
            )

            leg, bike_speed, run_pace, infeasible = compute_required_paces(
                km_after_swim,
                current_time_s,
                bike_speed_med,
                run_pace_med,
                target_stavsro_time_s,
            )
            RUN_ALERT_PACE_MIN_PER_KM = 5.5  # 5:30

            run_too_fast = (
                (run_pace is not None)
                and math.isfinite(run_pace)
                and (run_pace < RUN_ALERT_PACE_MIN_PER_KM)
            )


            col_bike, col_run = st.columns(2)
            with col_bike:
                st.metric("Bike", format_speed_kmh(bike_speed))
            with col_run:
                st.metric("Run", format_pace_min_per_km(run_pace))

            if leg == "pre_swim":
                st.info("Catch Up is intended for use after the swim (distance after swim ≥ 0 km).")
            elif leg == "beyond":
                st.info(
                    "Current distance is already beyond the Stavsro cut-off "
                    f"({TOTAL_STAVSRO_DIST_AFTER_SWIM:.1f} km after the swim)."
                )
            elif infeasible or run_too_fast:
                # If we are already on the run, warn specifically about the run pace.
                if leg == "run" or run_too_fast:
                    st.caption(
                        "⚠️ Note: to reach the target time, the required run pace is extremely fast "
                        "(faster than 5:30 min/km) – the value is mainly theoretical."
                    )
                else:
                    st.caption(
                        "⚠️ Note: to reach the target time, both bike and run would need to be much faster "
                        "than typical Black-Shirt performances – the required paces are purely theoretical."
                    )

        else:
            st.markdown("---")
            col_bike, col_run = st.columns(2)
            with col_bike:
                st.metric("Bike", "–")
            with col_run:
                st.metric("Run", "–")

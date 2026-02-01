import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bridge data layer 
try:
    import data_store
except Exception:
    data_store = None

# -------------------------------------------------------------------
# Konfiguration: Dateipfade & Spaltennamen
# -------------------------------------------------------------------

LONG_FILE = None  # provided via data_store.df_long() 
COURSE_PROFILE_FILE = None  # provided via data_store.course_profile() 

# Spaltennamen im Long-File
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
COL_CUM_TIME = "cum_time_seconds"  

# Wetter-Spalten
COL_TEMP = "temperature_2m"
COL_WIND_DIR = "wind_direction_10m"
COL_WIND_EFFECT = "wind_effect"
COL_WEATHER_EMOJI = "weather_emoji"

# Kursprofil
COL_COURSE_DIST = "distance_km"
COL_ELEV_NORSE = "elev_norseman_m"
COL_ELEV_HAWAII = "elev_hawaii_m"

# Ziel-Gruppen
GROUP_TOP10 = "Top 10"
GROUP_BLACK = "Black Shirt"
GROUP_WHITE = "White Shirt"
GROUP_DNF = "DNF"

GROUPS_ORDER = [GROUP_TOP10, GROUP_BLACK, GROUP_WHITE]
COL_GROUP = "group"

# -------------------------------------------------------------------
# Kanonische Split-Reihenfolge
# -------------------------------------------------------------------

ORDERED_SPLIT_KEYS = [
    "swim_finish",
    "transition_1_in",  # T1 (Event)
    "bike_start",       # Event
    "bike_6km_ovre_eidfjord",
    "bike_11km_enter_gamlevegen",
    "bike_16km_voringfossen",
    "bike_20km_garen",
    "bike_28km_bjoreio",
    "bike_36km_dyranut",
    "bike_47km_halne",
    "bike_66km_haugastol",
    "bike_90km_geilo",
    "bike_94km_kikut",
    "bike_101km_skurdalen",
    "bike_113km_dagali",
    "bike_123km_vasstulan",
    "bike_134km_start_imingfjell",
    "bike_142km_top_imingfjell",
    "bike_152km_end_imingfjell",
    "bike_180km_finish",
    "run_start",  # Event
    "run_5km_atraa",
    "run_10km",
    "run_15km_tinnsjo",
    "run_20km_miland",
    "run_25km_zombie_hill_base",
    "run_zombie_hill_1st_turn",
    "run_zombie_hill_2nd_turn",
    "run_zombie_hill_3rd_turn",
    "run_zombie_hill_4th_turn",
    "run_zombie_hill_5th_turn",
    "run_32_5km_langefonn",
    "run_37_5km_stavsro_cut_off",
    "run_40km_right_before_mt_gaustatoppen",
    "finish_black_t_shirt",
    "finish_white_t_shirt",
]

# -------------------------------------------------------------------
# Segment-Definitionen (explizit wie von dir beschrieben)
# -------------------------------------------------------------------

BIKE_CHECKPOINTS = [
    "bike_start",
    "bike_6km_ovre_eidfjord",
    "bike_11km_enter_gamlevegen",
    "bike_16km_voringfossen",
    "bike_20km_garen",
    "bike_28km_bjoreio",
    "bike_36km_dyranut",
    "bike_47km_halne",
    "bike_66km_haugastol",
    "bike_90km_geilo",
    "bike_94km_kikut",
    "bike_101km_skurdalen",
    "bike_113km_dagali",
    "bike_123km_vasstulan",
    "bike_134km_start_imingfjell",
    "bike_142km_top_imingfjell",
    "bike_152km_end_imingfjell",
    "bike_180km_finish",
]

RUN_CHECKPOINTS_BLACK = [
    "run_start",
    "run_5km_atraa",
    "run_10km",
    "run_15km_tinnsjo",
    "run_20km_miland",
    "run_25km_zombie_hill_base",
    "run_zombie_hill_1st_turn",
    "run_zombie_hill_2nd_turn",
    "run_zombie_hill_3rd_turn",
    "run_zombie_hill_4th_turn",
    "run_zombie_hill_5th_turn",
    "run_32_5km_langefonn",
    "run_37_5km_stavsro_cut_off",
    "run_40km_right_before_mt_gaustatoppen",
    "finish_black_t_shirt",
]

WHITE_FINISH_KEY = "finish_white_t_shirt"
WHITE_LAST_START_CANDIDATES = [
    "run_37_5km_stavsro_cut_off",
    "run_32_5km_langefonn",
]


def build_segment_pairs() -> list[dict]:
    segs = []

    # Swim: start -> swim_finish
    segs.append({"start": None, "end": "swim_finish", "leg": "swim", "end_label": "swim_finish"})

    # Bike
    for a, b in zip(BIKE_CHECKPOINTS[:-1], BIKE_CHECKPOINTS[1:]):
        segs.append({"start": a, "end": b, "leg": "bike", "end_label": b})

    # Run (Black)
    for a, b in zip(RUN_CHECKPOINTS_BLACK[:-1], RUN_CHECKPOINTS_BLACK[1:]):
        segs.append({"start": a, "end": b, "leg": "run", "end_label": b})

    # White dynamic final segment
    segs.append({"start": "WHITE_DYNAMIC", "end": WHITE_FINISH_KEY, "leg": "run", "end_label": WHITE_FINISH_KEY})
    return segs


SEGMENT_PAIRS = build_segment_pairs()

# -------------------------------------------------------------------
# Daten laden
# -------------------------------------------------------------------

@st.cache_data
def load_long_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_group_column(df)
    return df


@st.cache_data
def load_course_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(COL_COURSE_DIST)


# -------------------------------------------------------------------
# Group-Spalte
# -------------------------------------------------------------------

def ensure_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[COL_GROUP] = "Other"
    finish = df[COL_FINISH_TYPE].astype(str)

    df.loc[finish.str.upper().str.contains("DNF"), COL_GROUP] = GROUP_DNF

    mask_finisher = ~finish.str.upper().str.contains("DNF")

    df.loc[mask_finisher & finish.str.lower().str.contains("black"), COL_GROUP] = GROUP_BLACK
    df.loc[mask_finisher & finish.str.lower().str.contains("white"), COL_GROUP] = GROUP_WHITE

    ranks = pd.to_numeric(df[COL_OVERALL_RANK], errors="coerce")
    df.loc[mask_finisher & (ranks <= 10), COL_GROUP] = GROUP_TOP10

    return df


# -------------------------------------------------------------------
# Wind helpers
# -------------------------------------------------------------------

def wind_arrow_from_deg(deg: float) -> str:
    if pd.isna(deg):
        return "？"
    d = float(deg) % 360
    if 337.5 <= d or d < 22.5:
        return "↓"
    elif 22.5 <= d < 67.5:
        return "↙"
    elif 67.5 <= d < 112.5:
        return "←"
    elif 112.5 <= d < 157.5:
        return "↖"
    elif 157.5 <= d < 202.5:
        return "↑"
    elif 202.5 <= d < 247.5:
        return "↗"
    elif 247.5 <= d < 292.5:
        return "→"
    else:
        return "↘"


def wind_color_from_effect(effect) -> str:
    if pd.isna(effect):
        return "white"
    s = str(effect).lower()
    return "limegreen" if "tail" in s else "red"


# -------------------------------------------------------------------
# Weather aggregation (per split_key) + unique x positions 
# -------------------------------------------------------------------

def aggregate_weather(long_df: pd.DataFrame, year: int) -> pd.DataFrame:
    df_year = long_df[long_df[COL_YEAR] == year].copy()

    for col in [COL_DIST, COL_RACE_DIST, COL_TEMP, COL_WIND_DIR]:
        if col in df_year.columns:
            df_year[col] = pd.to_numeric(df_year[col], errors="coerce")

    df_year[COL_SPLIT_KEY] = df_year[COL_SPLIT_KEY].astype(str)
    df_year = df_year[df_year[COL_SPLIT_KEY].isin(ORDERED_SPLIT_KEYS)].copy()

    agg = (
        df_year
        .groupby(COL_SPLIT_KEY, as_index=False)
        .agg(
            {
                COL_RACE_DIST: "median",
                COL_DIST: "median",
                COL_TEMP: "median",
                COL_WIND_DIR: "median",
                COL_WIND_EFFECT: lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan,
                COL_WEATHER_EMOJI: lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan,
                COL_SPLIT_NAME: lambda x: x.mode().iloc[0] if len(x.mode()) else "",
            }
        )
    )

    agg["_split_order"] = pd.Categorical(agg[COL_SPLIT_KEY], ORDERED_SPLIT_KEYS, ordered=True)
    agg = agg.sort_values("_split_order").reset_index(drop=True)

    if not agg.empty:
        agg.loc[0, COL_WEATHER_EMOJI] = "🌊"

    # unique x if distances repeat (only for annotation placement)
    eps = 0.02
    agg[COL_RACE_DIST] = pd.to_numeric(agg[COL_RACE_DIST], errors="coerce")
    agg["_x"] = agg[COL_RACE_DIST].astype(float)
    agg["_dup_rank"] = agg.groupby("_x").cumcount()
    agg["_x"] = agg["_x"] + agg["_dup_rank"] * eps

    return agg


# -------------------------------------------------------------------
# Speed aggregation (explicit segment pairs)
# -------------------------------------------------------------------

def aggregate_speed_by_group(long_df: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {COL_YEAR, COL_BIB, COL_GROUP, COL_SPLIT_KEY, COL_RACE_DIST, COL_CUM_TIME}
    missing = required - set(long_df.columns)
    if missing:
        raise KeyError(f"aggregate_speed_by_group missing columns: {missing}")

    df = long_df[long_df[COL_YEAR] == year].copy()
    df[COL_SPLIT_KEY] = df[COL_SPLIT_KEY].astype(str)
    df = df[df[COL_SPLIT_KEY].isin(ORDERED_SPLIT_KEYS)].copy()

    df[COL_RACE_DIST] = pd.to_numeric(df[COL_RACE_DIST], errors="coerce")
    df[COL_CUM_TIME] = pd.to_numeric(df[COL_CUM_TIME], errors="coerce")
    df = df.dropna(subset=[COL_RACE_DIST, COL_CUM_TIME])

    # Kanonische Distanz pro split_key (median je Jahr)
    dist_map = df.groupby(COL_SPLIT_KEY)[COL_RACE_DIST].median()

    # Zeit pro Athlet pro split_key
    t = (
        df.groupby([COL_BIB, COL_GROUP, COL_SPLIT_KEY], as_index=False)[COL_CUM_TIME]
        .median()
    )
    t_piv = t.pivot_table(
        index=[COL_BIB, COL_GROUP],
        columns=COL_SPLIT_KEY,
        values=COL_CUM_TIME,
        aggfunc="median",
    )
    SWIM_KM_CAN = 3.8
    BIKE_KM_CAN = 180.0
    RUN_KM_CAN = 42.2

    def _parse_km_from_key(key: str) -> float | None:
        s = str(key)
        if "run_" in s:
            if s.startswith("run_") and "km" in s:
                mid = s.split("run_", 1)[1].split("km", 1)[0]
                mid = mid.replace("_", ".")
                try:
                    return float(mid)
                except Exception:
                    return None
        if s.startswith("bike_") and "km" in s:
            mid = s.split("bike_", 1)[1].split("km", 1)[0]
            mid = mid.replace("_", ".")
            try:
                return float(mid)
            except Exception:
                return None
        return None

    def canonical_cum_dist(key: str | None) -> float | None:
        if key is None:
            return 0.0
        k = str(key)
        if k == "swim_finish":
            return SWIM_KM_CAN
        if k in ("transition_1_in", "bike_start"):
            return SWIM_KM_CAN
        if k.startswith("bike_") and "km" in k:
            km = _parse_km_from_key(k)
            return None if km is None else (SWIM_KM_CAN + km)
        if k in ("bike_180km_finish", "run_start"):
            return SWIM_KM_CAN + BIKE_KM_CAN
        if k.startswith("run_") and "km" in k:
            km = _parse_km_from_key(k)
            return None if km is None else (SWIM_KM_CAN + BIKE_KM_CAN + km)
        if k in ("finish_black_t_shirt", "finish_white_t_shirt"):
            return SWIM_KM_CAN + BIKE_KM_CAN + RUN_KM_CAN
        return None

    def canonical_seg_distance(start: str | None, end: str | None) -> float | None:
        """Fallback segment length using canonical distances."""
        d0 = canonical_cum_dist(start)
        d1 = canonical_cum_dist(end)
        if d0 is None or d1 is None:
            return None
        dd = float(d1) - float(d0)
        return None if dd <= 0 else float(dd)

    precomputed_speed = {}
    if COL_SPEED in df.columns:
        df["_seg_speed_raw"] = pd.to_numeric(df[COL_SPEED], errors="coerce")
        precomputed_speed = (
            df.dropna(subset=["_seg_speed_raw"])
            .groupby([COL_GROUP, COL_SPLIT_KEY])["_seg_speed_raw"]
            .median()
            .to_dict()
        )
        df = df.drop(columns=["_seg_speed_raw"])

    rows = []

    for (bib, grp), times in t_piv.iterrows():

        def get_time(key: str | None) -> float | None:
            if key is None:
                return 0.0
            if key in times.index:
                v = times[key]
                return None if pd.isna(v) else float(v)
            return None

        def get_dist(key: str | None) -> float | None:
            if key is None:
                return 0.0
            if key in dist_map.index:
                v = dist_map[key]
                if not pd.isna(v):
                    return float(v)
            v2 = canonical_cum_dist(key)
            return None if v2 is None else float(v2)

        def resolve_white_start() -> str | None:
            # Priorität: 37.5 falls vorhanden, sonst 32.5
            for cand in WHITE_LAST_START_CANDIDATES:
                if get_time(cand) is not None:
                    return cand
            return None

        for seg in SEGMENT_PAIRS:
            start = seg["start"]
            end = seg["end"]
            leg = seg["leg"]

            # White dynamic segment nur für White Shirt berechnen
            if start == "WHITE_DYNAMIC":
                if grp != GROUP_WHITE:
                    continue
                start = resolve_white_start()
                if start is None:
                    continue

            # Black finish nicht für White
            if end == "finish_black_t_shirt" and grp == GROUP_WHITE:
                continue

            # White finish nur für White
            if end == WHITE_FINISH_KEY and grp != GROUP_WHITE:
                continue

            # If bike_start is missing for timing, use transition_1_in timing (same point in race)
            t0 = get_time(start)
            if (start == "bike_start") and (t0 is None):
                t0 = get_time("transition_1_in")

            t1 = get_time(end)

            d0 = get_dist(start)
            d1 = get_dist(end)

            if t0 is None or t1 is None or d0 is None or d1 is None:
                fallback_v = precomputed_speed.get((grp, end))
                if pd.notna(fallback_v):
                    dd_fallback = canonical_seg_distance(start, end)
                    rows.append(
                        {
                            COL_BIB: bib,
                            COL_GROUP: grp,
                            "segment_start_key": "race_start" if start is None else start,
                            "segment_end_key": end,
                            COL_SPEED: float(fallback_v),
                            "segment_time_s": np.nan,
                            "segment_dist_km": np.nan if dd_fallback is None else float(dd_fallback),
                        }
                    )
                continue

            dt = t1 - t0
            dd = d1 - d0

            if dt <= 0:
                fallback_v = precomputed_speed.get((grp, end))
                if pd.notna(fallback_v):
                    dd_fallback = canonical_seg_distance(start, end)
                    rows.append(
                        {
                            COL_BIB: bib,
                            COL_GROUP: grp,
                            "segment_start_key": "race_start" if start is None else start,
                            "segment_end_key": end,
                            COL_SPEED: float(fallback_v),
                            "segment_time_s": np.nan,
                            "segment_dist_km": np.nan if dd_fallback is None else float(dd_fallback),
                        }
                    )
                continue

            if dd <= 0.05:
                d0c = canonical_cum_dist(start)
                d1c = canonical_cum_dist(end)
                if (d0c is not None) and (d1c is not None):
                    dd_c = float(d1c) - float(d0c)
                    if dd_c > 0.05:
                        d0 = float(d0c)
                        d1 = float(d1c)
                        dd = dd_c

            v = np.nan
            if dd > 0.05:
                v = (dd / dt) * 3600.0

            if not pd.isna(v):
                if leg == "swim" and not (0.3 <= v <= 12):
                    fallback_v = precomputed_speed.get((grp, end))
                    if pd.notna(fallback_v):
                        v = float(fallback_v)
                    else:
                        continue
                if leg == "bike" and not (3 <= v <= 140):
                    fallback_v = precomputed_speed.get((grp, end))
                    if pd.notna(fallback_v):
                        v = float(fallback_v)
                    else:
                        continue
                if leg == "run" and not (2 <= v <= 35):
                    fallback_v = precomputed_speed.get((grp, end))
                    if pd.notna(fallback_v):
                        v = float(fallback_v)
                    else:
                        continue

            rows.append(
                {
                    COL_BIB: bib,
                    COL_GROUP: grp,
                    "segment_start_key": "race_start" if start is None else start,
                    "segment_end_key": end,
                    COL_SPEED: v,
                    "segment_time_s": float(dt),
                    "segment_dist_km": float(dd),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[COL_GROUP, "segment_start_key", "segment_end_key", COL_SPEED, "segment_time_s", "segment_dist_km"]
        )

    seg_df = pd.DataFrame(rows)

    # Median je Gruppe & Segment (Start->End)
    out = (
        seg_df.groupby([COL_GROUP, "segment_start_key", "segment_end_key"], as_index=False)
        .agg(
            {
                COL_SPEED: "median",
                "segment_time_s": "median",
                "segment_dist_km": "median",
            }
        )
    )

    # Sortierung nach End-Key (für saubere Reihenfolge im Plot)
    out["_split_order"] = pd.Categorical(out["segment_end_key"], ORDERED_SPLIT_KEYS, ordered=True)
    out = out.sort_values("_split_order").reset_index(drop=True)

    return out



# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

def build_weatherspeed_figure(
    course_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    speed_df: pd.DataFrame,
    groups_to_plot: list[str],
    split_dist_map: dict[str, float] | None = None,  
) -> go.Figure:

    # --- Ensure numeric ---
    course_df = course_df.copy()
    course_df[COL_COURSE_DIST] = pd.to_numeric(course_df[COL_COURSE_DIST], errors="coerce")
    course_df[COL_ELEV_NORSE] = pd.to_numeric(course_df[COL_ELEV_NORSE], errors="coerce")
    if COL_ELEV_HAWAII in course_df.columns:
        course_df[COL_ELEV_HAWAII] = pd.to_numeric(course_df[COL_ELEV_HAWAII], errors="coerce")

    weather_df = weather_df.copy()
    weather_df["_x"] = pd.to_numeric(weather_df.get("_x", np.nan), errors="coerce")
    weather_df[COL_RACE_DIST] = pd.to_numeric(weather_df.get(COL_RACE_DIST, np.nan), errors="coerce")
    weather_df[COL_SPLIT_KEY] = weather_df[COL_SPLIT_KEY].astype(str)

    # Build robust key->distance fallback
    if split_dist_map is None:
        split_dist_map = {}
        for _, r in weather_df.dropna(subset=[COL_RACE_DIST]).iterrows():
            split_dist_map[str(r[COL_SPLIT_KEY])] = float(r[COL_RACE_DIST])

    split_dist_map = {str(k): float(v) for k, v in split_dist_map.items() if not pd.isna(v)}
    SWIM_KM = 3.8
    BIKE_KM = 180.0
    RUN_KM = 42.2

    def _parse_km_from_key(key: str) -> float | None:
        s = str(key)
        if s.startswith("bike_") and "km" in s:
            mid = s.split("bike_", 1)[1].split("km", 1)[0].replace("_", ".")
            try:
                return float(mid)
            except Exception:
                return None
        if s.startswith("run_") and "km" in s:
            mid = s.split("run_", 1)[1].split("km", 1)[0].replace("_", ".")
            try:
                return float(mid)
            except Exception:
                return None
        return None

    split_dist_map.setdefault("race_start", 0.0)
    split_dist_map.setdefault("swim_finish", SWIM_KM)
    split_dist_map.setdefault("transition_1_in", SWIM_KM)
    split_dist_map.setdefault("bike_start", SWIM_KM)
    split_dist_map.setdefault("bike_180km_finish", SWIM_KM + BIKE_KM)
    split_dist_map.setdefault("run_start", SWIM_KM + BIKE_KM)
    split_dist_map.setdefault("finish_black_t_shirt", SWIM_KM + BIKE_KM + RUN_KM)
    split_dist_map.setdefault("finish_white_t_shirt", SWIM_KM + BIKE_KM + RUN_KM)

    for k in ORDERED_SPLIT_KEYS:
        kk = str(k)
        if kk.startswith("bike_") and "km" in kk:
            km = _parse_km_from_key(kk)
            if km is not None:
                split_dist_map.setdefault(kk, SWIM_KM + float(km))
        if kk.startswith("run_") and "km" in kk:
            km = _parse_km_from_key(kk)
            if km is not None:
                split_dist_map.setdefault(kk, SWIM_KM + BIKE_KM + float(km))

    split_dist_map["race_start"] = 0.0

    key_to_x_weather = {}
    for _, r in weather_df.dropna(subset=["_x"]).iterrows():
        key_to_x_weather[str(r[COL_SPLIT_KEY])] = float(r["_x"])
    key_to_x_weather["race_start"] = 0.0

    # ---------------------------------------------------------------
    # Leg-relative X-Achse (Swim 0-3.8, Bike 0-180, Run 0-42.2)
    # ---------------------------------------------------------------
    GAP_KM = 0.0  

    swim_end = SWIM_KM
    bike_end = SWIM_KM + BIKE_KM

    bike_offset = SWIM_KM + GAP_KM
    run_offset = (SWIM_KM + GAP_KM) + BIKE_KM + GAP_KM

    def _x_transform(cum_dist: float) -> float:
        if pd.isna(cum_dist):
            return np.nan
        x = float(cum_dist)
        if x <= swim_end + 1e-9:
            return x
        if x <= bike_end + 1e-9:
            return (x - swim_end) + bike_offset
        return (x - bike_end) + run_offset

    x_max = run_offset + RUN_KM

    def x_of(key: str) -> float | None:
        k = str(key)
        if k in key_to_x_weather:
            return _x_transform(float(key_to_x_weather[k]))
        if k in split_dist_map:
            return _x_transform(float(split_dist_map[k]))
        return None

    def leg_km_label_from_key(key: str) -> str | None:
        k = str(key)
        if k not in split_dist_map:
            return None
        d = float(split_dist_map[k])
        if d <= swim_end + 1e-9:
            return f"{d:g}"
        if d <= bike_end + 1e-9:
            return f"{(d - swim_end):g}"
        return f"{(d - bike_end):g}"

    fig = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1, shared_xaxes=True)

    # ---------------------------------------------------------------
    # Elevation (FORCED BACKGROUND)
    # ---------------------------------------------------------------
    x_course = course_df[COL_COURSE_DIST].apply(lambda v: _x_transform(v))

    fig.add_trace(
        go.Scatter(
            x=x_course,
            y=course_df[COL_ELEV_NORSE],
            fill="tozeroy",
            name="Norseman elevation",
            mode="lines",
            line=dict(width=1),
            fillcolor="rgba(255,255,255,0.18)",
            opacity=0.55,
            hovertemplate="km %{x:.1f}<br>Elev: %{y:.0f} m<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )

    if COL_ELEV_HAWAII in course_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_course,
                y=course_df[COL_ELEV_HAWAII],
                name="Ironman Hawaii elevation",
                mode="lines",
                line=dict(width=2, dash="dash"),
                opacity=0.99,
                hovertemplate="Hawaii km %{x:.1f}<br>Elev: %{y:.0f} m<extra></extra>",
            ),
            row=1, col=1, secondary_y=True,
        )

    # ---------------------------------------------------------------
    # Vertical split lines (use robust distance positions)
    # ---------------------------------------------------------------
    split_positions = []
    for k in ORDERED_SPLIT_KEYS:
        xv = x_of(k)
        if xv is not None:
            split_positions.append(float(xv))

    split_positions = sorted(split_positions)
    dedup = []
    for v in split_positions:
        if not dedup or abs(v - dedup[-1]) > 0.01:
            dedup.append(v)
    split_positions = dedup

    for x in split_positions:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x, x1=x,
            y0=0, y1=1,
            line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
            layer="below",
        )

    # ---------------------------------------------------------------
    # Orange markers at swim_finish and run_start
    # ---------------------------------------------------------------
    for key in ["swim_finish", "run_start"]:
        xv = x_of(key)
        if xv is None:
            continue
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=float(xv), x1=float(xv),
            y0=0, y1=1,
            line=dict(color="orange", width=2, dash="dash"),
            layer="above",
        )

    # ---------------------------------------------------------------
    # Weather annotations (sparse)
    # ---------------------------------------------------------------
    weather_sorted = weather_df.dropna(subset=[COL_RACE_DIST]).sort_values(COL_RACE_DIST).reset_index(drop=True)

    keep_idx = []
    last_real = None
    for i, r in weather_sorted.iterrows():
        d = float(r[COL_RACE_DIST])
        if last_real is None or (d - last_real) >= 10.0:
            keep_idx.append(i)
            last_real = d
    weather_sparse = weather_sorted.loc[keep_idx].copy()

    Y_ARROW = 1.16
    Y_EMOJI = 1.09
    Y_TEMP = 1.03

    ARROW_SIZE = 24
    EMOJI_SIZE = 32
    TEMP_SIZE = 16

    for _, r in weather_sparse.iterrows():
        key = str(r[COL_SPLIT_KEY])
        x = x_of(key)
        if x is None:
            continue

        temp = r.get(COL_TEMP, np.nan)
        wdir = r.get(COL_WIND_DIR, np.nan)
        weffect = r.get(COL_WIND_EFFECT, np.nan)
        emoji = r.get(COL_WEATHER_EMOJI, "❔")
        emoji = emoji if isinstance(emoji, str) else "❔"

        arrow = wind_arrow_from_deg(wdir)
        arrow_color = wind_color_from_effect(weffect)

        show_arrow = (arrow != "？") and (key != "swim_finish")

        if show_arrow:
            fig.add_annotation(
                x=float(x), y=Y_ARROW, xref="x", yref="paper",
                text=arrow, showarrow=False,
                font=dict(size=ARROW_SIZE, color=arrow_color),
                xanchor="center", yanchor="middle",
            )

        fig.add_annotation(
            x=float(x), y=Y_EMOJI, xref="x", yref="paper",
            text=emoji, showarrow=False,
            font=dict(size=EMOJI_SIZE),
            xanchor="center", yanchor="middle",
        )

        fig.add_annotation(
            x=float(x), y=Y_TEMP, xref="x", yref="paper",
            text=f"{float(temp):.1f}°C" if not pd.isna(temp) else "–",
            showarrow=False,
            font=dict(size=TEMP_SIZE),
            xanchor="center", yanchor="middle",
        )

    # ---------------------------------------------------------------
    # Speed segments (horizontal per split segment)
    # ---------------------------------------------------------------
    color_map = {
        GROUP_TOP10: "#00ff00",
        GROUP_BLACK: "#000000",
        GROUP_WHITE: "#ffffff",
    }

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(symbol="arrow-up", color="limegreen", size=14, line=dict(color="white", width=1)),
            showlegend=True,
            name="Wind arrow (green = tailwind)",
            hoverinfo="skip",
            opacity=0.01,  
        ),
        row=1, col=1, secondary_y=False,
    )

    preferred_order = [GROUP_WHITE, GROUP_BLACK, GROUP_TOP10]
    plot_groups = sorted(
        groups_to_plot,
        key=lambda g: preferred_order.index(g) if g in preferred_order else len(preferred_order),
    )

    group_offsets = {
        GROUP_TOP10: 0.35,
        GROUP_BLACK: -0.35,
        GROUP_WHITE: 0.0,
    }

    for group in plot_groups:
        df_g = speed_df[speed_df[COL_GROUP] == group].copy()
        if df_g.empty:
            continue

        df_g["_ord"] = pd.Categorical(df_g["segment_end_key"], ORDERED_SPLIT_KEYS, ordered=True)
        df_g = df_g.sort_values("_ord").reset_index(drop=True)

        x_line, y_line, hover_vals = [], [], []
        y_offset = group_offsets.get(group, 0.0)

        for _, r in df_g.iterrows():
            sk = str(r["segment_start_key"])
            ek = str(r["segment_end_key"])

            x0 = x_of(sk)
            x1 = x_of(ek)
            if x0 is None or x1 is None:
                continue

            if abs(x1 - x0) < 1e-6:
                x1 = x0 + 0.02

            v = r.get(COL_SPEED, np.nan)
            if pd.isna(v):
                continue

            v_real = float(v)
            y = v_real + y_offset

            x_line += [float(x0), float(x1), None]
            y_line += [y, y, None]
            hover_vals += [v_real, v_real, None]

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                customdata=hover_vals,
                mode="lines",
                name=f"{group} median speed",
                line=dict(width=5, color=color_map.get(group, None)),
                hovertemplate=f"{group}<br>v = %{{customdata:.1f}} km/h<extra></extra>",
            ),
            row=1, col=1, secondary_y=False,
        )

    # ---------------------------------------------------------------
    # FORCE LAYERING: Elevation behind speed (trace order)
    # ---------------------------------------------------------------
    elev_names = {"Norseman elevation", "Ironman Hawaii elevation"}
    elev_traces = [tr for tr in fig.data if getattr(tr, "name", None) in elev_names]
    other_traces = [tr for tr in fig.data if getattr(tr, "name", None) not in elev_names]
    fig.data = tuple(elev_traces + other_traces)

    # ---------------------------------------------------------------
    # Axes & layout
    # ---------------------------------------------------------------
    speed_ticks = list(range(0, 71, 10))  # 0,10,...,70  -> 7 ticks (7 Intervalle)

    fig.update_yaxes(
        title_text="Speed (km/h)",
        secondary_y=False,
        range=[0, 70],
        tickmode="array",
        tickvals=speed_ticks,
        showgrid=True,
        zeroline=False,
        title_font=dict(size=22),
        tickfont=dict(size=14),
    )

    # --- RIGHT (Elevation): NO gridlines at all ---
    fig.update_yaxes(
        title_text="Elevation (m)",
        secondary_y=True,
        side="right",
        rangemode="tozero",
        showgrid=False,
        zeroline=False,
        title_font=dict(size=22),
        tickfont=dict(size=14),
    )

    # Major ticks (leg ends) + additional informative split ticks (Bike 6,36,66,90,123,152 | Run 10,25,37.5)
    tickvals = [
        0.0,
        SWIM_KM,
        bike_offset + BIKE_KM,
        run_offset + RUN_KM,
    ]
    ticktext = ["0", f"{SWIM_KM:g}", f"{BIKE_KM:g}", f"{RUN_KM:g}"]

    extra_keys = [
        "bike_6km_ovre_eidfjord",
        "bike_36km_dyranut",
        "bike_66km_haugastol",
        "bike_90km_geilo",
        "bike_123km_vasstulan",
        "bike_152km_end_imingfjell",
        "run_10km",
        "run_25km_zombie_hill_base",
        "run_37_5km_stavsro_cut_off",
    ]
    for k in extra_keys:
        xv = x_of(k)
        lab = leg_km_label_from_key(k)
        if xv is None or lab is None:
            continue
        tickvals.append(float(xv))
        ticktext.append(lab)

    # stable tick ordering
    pairs = sorted(zip(tickvals, ticktext), key=lambda p: p[0])
    tickvals = [p[0] for p in pairs]
    ticktext = [p[1] for p in pairs]

    fig.update_xaxes(
        title_text="Distance (km)",
        range=[0, x_max],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title_font=dict(size=22),
        tickfont=dict(size=14),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#7A7A7A",
        plot_bgcolor="#7A7A7A",
        margin=dict(l=50, r=120, t=110, b=60),
        height=560,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=28),
        ),
        hovermode="x unified",
        annotations=list(fig.layout.annotations) + [
            dict(
                x=SWIM_KM / 2,
                y=1.26,
                xref="x",
                yref="paper",
                text="Swim",
                showarrow=False,
                font=dict(color="white", size=20),
                xanchor="center",
            ),
            dict(
                x=bike_offset + BIKE_KM / 2,
                y=1.26,
                xref="x",
                yref="paper",
                text="Bike",
                showarrow=False,
                font=dict(color="white", size=20),
                xanchor="center",
            ),
            dict(
                x=run_offset + RUN_KM / 2,
                y=1.26,
                xref="x",
                yref="paper",
                text="Run",
                showarrow=False,
                font=dict(color="white", size=20),
                xanchor="center",
            ),
        ],
    )

    return fig




# -------------------------------------------------------------------
# Streamlit entry
# -------------------------------------------------------------------
def render_weatherspeed(selected_year="All", selected_group="All"):
    col_title2, col_info2 = st.columns([0.92, 0.08], vertical_alignment="center")

    with col_title2:
        st.header("Course, Weather & Speed")

    with col_info2:
        with st.popover("ℹ️"):
               st.markdown(
            """
**What is shown in this chart?**

This visualization combines **course profile + weather + group median speed** for the selected year.

- **X-axis**: course distance, split into **Swim (0–3.8 km)**, **Bike (0–180 km)**, **Run (0–42.2 km)**  
- **Left Y-axis**: **speed** in km/h (horizontal segments)
- **Right Y-axis**: **elevation** in meters (filled background area)

- **Speed lines**: median segment speed per finish group (**Top 10 / Black Shirt / White Shirt**)  
- **Weather markers**: sparse icons along the course showing **temperature**, **weather emoji**, and **wind direction**  
  (wind arrow color: **green = tailwind**, **red = headwind/crosswind**)

  **How to read it**
- Each **horizontal segment** is the median speed **between two checkpoints** (split → next split).
- **Higher segments** mean **faster** group speed in that section.
- Sudden **drops** often coincide with **climbs**, technical terrain, or fatigue later in the race.
- Weather icons let you spot where **wind/temperature changes** may align with speed changes.

**How this can be used**
- Compare **Top 10 vs. Black/White** to see **where gaps open up** (e.g., climbs, exposed windy sections).
- Identify segments where speed changes most across groups → potential **decisive race sections**.
- Use wind arrows to interpret “slow” bike segments: **headwind sections** often flatten speeds across groups.
- Relate speed dips to the elevation profile to distinguish **terrain impact vs. pacing**.
"""
        )

    # --------------------------------------------------
    # Load data via Bridge (multipage-safe)
    # --------------------------------------------------
    try:
        import data_store
        long_df = data_store.df_long()
        course_df = data_store.course_profile()
    except Exception as e:
        st.error(
            "Konnte die Daten nicht über data_store laden.\n\n"
            f"Fehler: {e}"
        )
        return

    # --------------------------------------------------
    # Ensure COL_GROUP exists using the canonical logic in this file
    # --------------------------------------------------
    if COL_GROUP not in long_df.columns:
        needed = {COL_FINISH_TYPE, COL_OVERALL_RANK}
        if not needed.issubset(long_df.columns):
            st.error(
                f"Cannot build '{COL_GROUP}'. Missing columns: {needed - set(long_df.columns)}\n\n"
                f"Available columns: {list(long_df.columns)}"
            )
            return
        long_df = ensure_group_column(long_df)

    # --------------------------------------------------
    # Year selection (from header)
    # --------------------------------------------------
    if selected_year == "All":
        year = int(pd.to_numeric(long_df[COL_YEAR], errors="coerce").max())
    else:
        year = int(selected_year)

    # --------------------------------------------------
    # Group selection (from header)
    # --------------------------------------------------
    available_groups = sorted(long_df[COL_GROUP].dropna().unique())

    if selected_group == "All":
        groups_to_plot = [g for g in GROUPS_ORDER if g in available_groups]
    elif selected_group == GROUP_DNF:
        groups_to_plot = []
    else:
        groups_to_plot = [selected_group] if selected_group in available_groups else []

    # --------------------------------------------------
    # Apply group filter early for speed calculation (makes the filter effect explicit)
    # --------------------------------------------------
    long_df_for_speed = (
        long_df[long_df[COL_GROUP].isin(groups_to_plot)].copy()
        if groups_to_plot
        else long_df.iloc[0:0].copy()
    )

    # --------------------------------------------------
    # Robust split_dist_map from the actual long_df (year-specific)
    # --------------------------------------------------
    df_year = long_df[long_df[COL_YEAR] == year].copy()
    df_year[COL_SPLIT_KEY] = df_year[COL_SPLIT_KEY].astype(str)
    df_year[COL_RACE_DIST] = pd.to_numeric(df_year[COL_RACE_DIST], errors="coerce")
    df_year = df_year[df_year[COL_SPLIT_KEY].isin(ORDERED_SPLIT_KEYS)].dropna(subset=[COL_RACE_DIST])

    split_dist_map = (
        df_year.groupby(COL_SPLIT_KEY)[COL_RACE_DIST]
        .median()
        .to_dict()
    )
    split_dist_map = {str(k): float(v) for k, v in split_dist_map.items() if not pd.isna(v)}
    split_dist_map["race_start"] = 0.0

    SWIM_KM_CAN = 3.8
    BIKE_KM_CAN = 180.0
    RUN_KM_CAN = 42.2

    def _parse_km_from_key(key: str) -> float | None:
        s = str(key)
        if s.startswith("bike_") and "km" in s:
            mid = s.split("bike_", 1)[1].split("km", 1)[0]
            mid = mid.replace("_", ".")
            try:
                return float(mid)
            except Exception:
                return None
        if s.startswith("run_") and "km" in s:
            mid = s.split("run_", 1)[1].split("km", 1)[0]
            mid = mid.replace("_", ".")
            try:
                return float(mid)
            except Exception:
                return None
        return None

    # Always enforce leg anchor points
    split_dist_map["race_start"] = 0.0
    split_dist_map["swim_finish"] = SWIM_KM_CAN
    split_dist_map["transition_1_in"] = SWIM_KM_CAN
    split_dist_map["bike_start"] = SWIM_KM_CAN
    split_dist_map["bike_180km_finish"] = SWIM_KM_CAN + BIKE_KM_CAN
    split_dist_map["run_start"] = SWIM_KM_CAN + BIKE_KM_CAN
    split_dist_map["finish_black_t_shirt"] = SWIM_KM_CAN + BIKE_KM_CAN + RUN_KM_CAN
    split_dist_map["finish_white_t_shirt"] = SWIM_KM_CAN + BIKE_KM_CAN + RUN_KM_CAN

    # Fill canonical distances for ALL expected split keys (not only those present in df_year)
    for k in ORDERED_SPLIT_KEYS:
        kk = str(k)
        if kk.startswith("bike_") and "km" in kk:
            km = _parse_km_from_key(kk)
            if km is not None:
                split_dist_map[kk] = SWIM_KM_CAN + float(km)
        if kk.startswith("run_") and "km" in kk:
            km = _parse_km_from_key(kk)
            if km is not None:
                split_dist_map[kk] = SWIM_KM_CAN + BIKE_KM_CAN + float(km)

    # --------------------------------------------------
    # Aggregate weather + speed
    # --------------------------------------------------
    weather_df = aggregate_weather(long_df, year)
    speed_df_all = aggregate_speed_by_group(long_df_for_speed, year)

    if groups_to_plot:
        speed_df = speed_df_all[speed_df_all[COL_GROUP].isin(groups_to_plot)].copy()
    else:
        speed_df = speed_df_all.iloc[0:0].copy()

    # --------------------------------------------------
    # Build + render figure
    # --------------------------------------------------
    fig = build_weatherspeed_figure(
        course_df,
        weather_df,
        speed_df,
        groups_to_plot,
        split_dist_map=split_dist_map,
    )
    st.plotly_chart(fig, width="stretch")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.race_logic import (
    apply_group_filter,
    apply_year_filter,
    coerce_year,
    get_group_color,
    get_named_color,
)

from data_store import course_profile as ds_course_profile
from data_store import df_model as ds_df_model
from data_store import df_long as ds_df_long
from modules import accumulated as accumulated_module

TARGET_PROB = 0.50
PANEL_BG = "#7a7a7a"
FRONTIER_COLOR = get_named_color("frontier_pace")
SWIM_KM = 3.8

ORDERED_SPLIT_KEYS = [
    "swim_finish",
    "transition_1_in",
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
]


def _format_hms(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    s = int(round(float(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:d}:{m:02d}:{sec:02d}"


def _cutoff_total_km_for_year(year_value) -> float:
    y = coerce_year(year_value)
    if y in {2024, 2025}:
        return 221.3
    return 216.3


def _load_model_base() -> pd.DataFrame:
    df = ds_df_model().copy()
    if "set" in df.columns:
        df = df[df["set"] == "test"].copy()

    numeric_cols = ["year", "race_distance_km", "cum_time_seconds", "p_black", "segment_speed_kmh", "split_rank"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = ["split_key", "race_distance_km", "cum_time_seconds", "p_black"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in model data: {missing}")

    df["split_key"] = df["split_key"].astype(str)
    df = df.dropna(subset=["split_key", "race_distance_km", "cum_time_seconds", "p_black"])
    return df


def _frontier_points(df: pd.DataFrame) -> pd.DataFrame:
    def infer_leg(split_key: str) -> str:
        k = str(split_key)
        if k.startswith("swim"):
            return "swim"
        if k.startswith("bike"):
            return "bike"
        if k.startswith("run"):
            return "run"
        if k in {"transition_1_in", "bike_start"}:
            return "transition"
        return "other"

    def interpolate_frontier_time(split_df: pd.DataFrame) -> tuple[float, float, str]:
        """
        Returns (frontier_time_seconds, frontier_prob, method).
        method: 'interpolated' or 'nearest'
        """
        s = (
            split_df[["cum_time_seconds", "p_black"]]
            .dropna()
            .groupby("cum_time_seconds", as_index=False)["p_black"]
            .median()
            .sort_values("cum_time_seconds")
            .reset_index(drop=True)
        )
        if s.empty:
            return np.nan, np.nan, "nearest"

        t = s["cum_time_seconds"].astype(float).to_numpy()
        p = s["p_black"].astype(float).to_numpy()
        d = p - TARGET_PROB

        # Exact hit
        exact_idx = np.where(np.isclose(d, 0.0))[0]
        if len(exact_idx) > 0:
            i = int(exact_idx[0])
            return float(t[i]), TARGET_PROB, "interpolated"

        # Candidate sign changes between adjacent points
        candidates = []
        for i in range(len(d) - 1):
            d1, d2 = float(d[i]), float(d[i + 1])
            if d1 * d2 < 0:
                p1, p2 = float(p[i]), float(p[i + 1])
                t1, t2 = float(t[i]), float(t[i + 1])
                if p2 == p1:
                    continue
                t_star = t1 + (TARGET_PROB - p1) * (t2 - t1) / (p2 - p1)
                quality = (abs(d1) + abs(d2)) / 2.0
                candidates.append((quality, t_star))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return float(candidates[0][1]), TARGET_PROB, "interpolated"

        # Fallback: no crossing in this split
        idx = int(np.argmin(np.abs(d)))
        return float(t[idx]), float(p[idx]), "nearest"

    rows = []
    grouped = df.groupby("split_key", as_index=False)

    for split_key, g in grouped:
        g = g.copy()
        if g.empty:
            continue

        frontier_t, frontier_p, method = interpolate_frontier_time(g)
        if not np.isfinite(frontier_t):
            continue

        # Segment speed close to frontier time
        g_speed = g.copy()
        if "segment_speed_kmh" in g_speed.columns:
            g_speed["dt_to_frontier"] = (g_speed["cum_time_seconds"] - frontier_t).abs()
            g_speed = g_speed.sort_values("dt_to_frontier")
            k = max(5, int(round(len(g_speed) * 0.10)))
            k = min(k, len(g_speed))
            gk = g_speed.head(k)
            speed_raw = float(gk["segment_speed_kmh"].median()) if gk["segment_speed_kmh"].notna().any() else np.nan
        else:
            speed_raw = np.nan

        rows.append(
            {
                "split_key": split_key,
                "race_distance_km": float(g["race_distance_km"].median()),
                "cum_time_seconds": float(frontier_t),
                "p_black_frontier": float(frontier_p),
                "frontier_method": method,
                "segment_speed_kmh_raw": speed_raw,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["_ord"] = pd.Categorical(out["split_key"], categories=ORDERED_SPLIT_KEYS, ordered=True)
    out = out.sort_values(["_ord", "race_distance_km", "cum_time_seconds"]).reset_index(drop=True)
    out["cum_time_seconds"] = out["cum_time_seconds"].cummax()
    out["leg"] = out["split_key"].map(infer_leg)
    out["segment_dist_km"] = out["race_distance_km"].diff().fillna(out["race_distance_km"])
    out["segment_time_s"] = out["cum_time_seconds"].diff().fillna(out["cum_time_seconds"])
    out.loc[out["segment_time_s"] <= 0, "segment_time_s"] = np.nan
    out.loc[out["segment_dist_km"] <= 0, "segment_dist_km"] = np.nan

    # Swim segment must be computed from swim finish time (race start -> swim_finish).
    swim_mask = out["split_key"].eq("swim_finish")
    out.loc[swim_mask, "segment_dist_km"] = SWIM_KM
    out.loc[swim_mask, "segment_time_s"] = out.loc[swim_mask, "cum_time_seconds"]

    out["segment_speed_kmh_from_time"] = (out["segment_dist_km"] / out["segment_time_s"]) * 3600.0

    # Use time-derived speed as primary, model speed as fallback.
    out["segment_speed_kmh"] = out["segment_speed_kmh_from_time"].where(
        out["segment_speed_kmh_from_time"].notna(),
        out["segment_speed_kmh_raw"],
    )

    # Plausibility bounds per leg.
    swim_bad = out["leg"].eq("swim") & ((out["segment_speed_kmh"] < 1.0) | (out["segment_speed_kmh"] > 6.5))
    bike_bad = out["leg"].eq("bike") & ((out["segment_speed_kmh"] < 5.0) | (out["segment_speed_kmh"] > 65.0))
    run_bad = out["leg"].eq("run") & ((out["segment_speed_kmh"] < 3.0) | (out["segment_speed_kmh"] > 25.0))
    out.loc[swim_bad | bike_bad | run_bad, "segment_speed_kmh"] = out.loc[swim_bad | bike_bad | run_bad, "segment_speed_kmh_raw"]
    out.loc[(out["segment_speed_kmh"] <= 0) | (out["segment_speed_kmh"] > 80), "segment_speed_kmh"] = np.nan

    out = out.drop(columns=["_ord"])
    return out


def _render_frontier_table(frontier_df: pd.DataFrame) -> None:
    if frontier_df.empty:
        st.info("No frontier data available for the current filters.")
        return

    show = frontier_df.copy()
    show["cum_time"] = show["cum_time_seconds"].apply(_format_hms)
    show["segment_time"] = show["segment_time_s"].apply(_format_hms)
    show["distance_km"] = show["race_distance_km"].round(1)
    show["frontier_prob"] = (show["p_black_frontier"] * 100.0).round(1).astype(str) + " %"
    show["segment_speed_kmh"] = show["segment_speed_kmh"].round(1)
    show["method"] = show["frontier_method"].str.replace("interpolated", "interp", regex=False).str.replace("nearest", "fallback", regex=False)

    out = show[["split_key", "distance_km", "segment_time", "cum_time", "segment_speed_kmh"]].rename(
        columns={
            "split_key": "Split",
            "distance_km": "Distance (km)",
            "segment_time": "Segment time",
            "cum_time": "Cumulative time",
            "segment_speed_kmh": "Frontier speed (km/h)",
        }
    )
    st.dataframe(out, use_container_width=True, hide_index=True)

    total_secs = float(frontier_df["cum_time_seconds"].iloc[-1])
    st.caption(f"Estimated frontier total at last split: **{_format_hms(total_secs)}**")


def _render_frontier_chart(frontier_df: pd.DataFrame, course_df: pd.DataFrame, cutoff_km: float) -> None:
    fig = go.Figure()

    c = course_df.copy()
    c = c[(c["distance_km"] >= 0) & (c["distance_km"] <= cutoff_km)]
    c["distance_km"] = pd.to_numeric(c["distance_km"], errors="coerce")
    c["elev_norseman_m"] = pd.to_numeric(c["elev_norseman_m"], errors="coerce")
    c = c.dropna(subset=["distance_km", "elev_norseman_m"])

    fig.add_trace(
        go.Scatter(
            x=c["distance_km"],
            y=c["elev_norseman_m"],
            mode="lines",
            name="Norseman elevation",
            line=dict(width=2, color="rgba(31,111,178,0.6)"),
            fill="tozeroy",
            fillcolor="rgba(31,111,178,0.18)",
            yaxis="y2",
            hovertemplate="km %{x:.1f}<br>Elev %{y:.0f} m<extra></extra>",
        )
    )

    x_vals, y_vals = [], []
    for i in range(len(frontier_df)):
        # First segment represents Swim from race start (0 km) to first split.
        if i == 0:
            x0 = 0.0
            x1 = float(frontier_df["race_distance_km"].iloc[i])
        else:
            x0 = float(frontier_df["race_distance_km"].iloc[i - 1])
            x1 = float(frontier_df["race_distance_km"].iloc[i])

        # Skip zero-distance transitions in speed profile.
        if (x1 - x0) <= 0.05:
            continue

        v = frontier_df["segment_speed_kmh"].iloc[i]
        if not np.isfinite(v):
            continue
        x_vals.extend([x0, x1, np.nan])
        y_vals.extend([float(v), float(v), np.nan])

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="Frontier pace (P=50%)",
            line=dict(width=6, color=FRONTIER_COLOR),
            hovertemplate="Frontier speed: %{y:.1f} km/h<extra></extra>",
        )
    )

    fig.add_vline(
        x=cutoff_km,
        line_dash="dash",
        line_width=1,
        annotation_text="Cut-off distance",
        annotation_position="top",
    )

    # Show split positions as dotted guides.
    split_x = sorted(set(np.round(frontier_df["race_distance_km"].dropna().astype(float), 3)))
    for x in split_x:
        if 0 <= x <= cutoff_km:
            fig.add_vline(
                x=float(x),
                line_dash="dot",
                line_width=1,
                line_color="rgba(220,220,220,0.45)",
            )

    fig.update_layout(
        xaxis=dict(title="Distance (km)", range=[0, cutoff_km]),
        yaxis=dict(title="Frontier speed (km/h)"),
        yaxis2=dict(title="Elevation (m)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _x_transform_accumulated(cum_dist_km: float) -> float:
    swim_km = 3.8
    bike_km = 180.0
    gap_km = 6.0
    bike_end = swim_km + bike_km
    bike_offset = swim_km + gap_km
    run_offset = (swim_km + gap_km) + bike_km + gap_km

    x = float(cum_dist_km)
    if x <= swim_km + 1e-9:
        return x
    if x <= bike_end + 1e-9:
        return (x - swim_km) + bike_offset
    return (x - bike_end) + run_offset


def _render_frontier_on_accumulated(selected_year, selected_group, frontier_df: pd.DataFrame) -> None:
    relative_mode = st.toggle("Relative to Frontier", value=False, key="frontier_relative_toggle")

    long_df = ds_df_long().copy()
    long_df = apply_year_filter(long_df, selected_year, year_col="year")
    long_df = apply_group_filter(
        long_df,
        selected_group,
        finish_col="finish_type",
        rank_col="overall_rank",
        top10_col="Top10_flag",
    )

    if long_df.empty:
        st.info("No accumulated race data for current filters.")
        return

    if relative_mode:
        _render_relative_to_frontier(long_df, frontier_df)
        return

    fig = accumulated_module._create_accumulated_figure(long_df, selected_group=str(selected_group))

    f = frontier_df.copy()
    f["x_plot"] = f["race_distance_km"].astype(float).apply(_x_transform_accumulated)
    f["cum_time_hours"] = f["cum_time_seconds"].astype(float) / 3600.0

    fig.add_trace(
        go.Scatter(
            x=f["x_plot"],
            y=f["cum_time_hours"],
            mode="lines+markers",
            name="Frontier time (P=50%)",
            line=dict(color=FRONTIER_COLOR, width=4),
            marker=dict(size=6, color=FRONTIER_COLOR),
            hovertemplate="Frontier<br>km %{x:.1f}<br>time %{y:.2f} h<extra></extra>",
            showlegend=True,
        )
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_relative_to_frontier(long_df: pd.DataFrame, frontier_df: pd.DataFrame) -> None:
    required = {"bib", "split_key", "cum_time_seconds"}
    if not required.issubset(long_df.columns):
        st.info("Relative frontier view requires bib, split_key, cum_time_seconds.")
        return

    df = long_df.copy()
    df["split_key"] = df["split_key"].astype(str)
    df["cum_time_seconds"] = pd.to_numeric(df["cum_time_seconds"], errors="coerce")
    df = df.dropna(subset=["bib", "split_key", "cum_time_seconds"])
    if df.empty:
        st.info("No data for relative frontier view.")
        return

    # Use median athlete time per split if duplicates exist for the same athlete/split.
    athlete_split = (
        df.groupby(["bib", "split_key"], as_index=False)["cum_time_seconds"]
        .median()
    )

    f = frontier_df[["split_key", "cum_time_seconds", "race_distance_km"]].copy()
    f = f.rename(columns={"cum_time_seconds": "frontier_time_s"})
    f["split_key"] = f["split_key"].astype(str)
    f["frontier_time_s"] = pd.to_numeric(f["frontier_time_s"], errors="coerce")
    f["race_distance_km"] = pd.to_numeric(f["race_distance_km"], errors="coerce")
    f = f.dropna(subset=["split_key", "frontier_time_s", "race_distance_km"])

    merged = athlete_split.merge(f, on="split_key", how="inner")
    if merged.empty:
        st.info("No overlapping splits between athlete data and frontier.")
        return

    merged["x_distance_km"] = merged["race_distance_km"].astype(float)
    merged["delta_h"] = (merged["cum_time_seconds"] - merged["frontier_time_s"]) / 3600.0

    # Build per-athlete group label (Top 10 / Black Shirt / White Shirt / DNF).
    meta_cols = ["bib"]
    if "finish_type" in df.columns:
        meta_cols.append("finish_type")
    if "overall_rank" in df.columns:
        meta_cols.append("overall_rank")
    if "name" in df.columns:
        meta_cols.append("name")

    athlete_meta = df[meta_cols].drop_duplicates(subset=["bib"]).copy()
    if "finish_type" not in athlete_meta.columns:
        athlete_meta["finish_type"] = ""
    if "overall_rank" not in athlete_meta.columns:
        athlete_meta["overall_rank"] = np.nan

    finish = athlete_meta["finish_type"].astype(str).str.lower()
    rank = pd.to_numeric(athlete_meta["overall_rank"], errors="coerce")
    athlete_meta["group_label"] = "White Shirt"
    athlete_meta.loc[finish.str.contains("dnf|did not finish|dns", regex=True, na=False), "group_label"] = "DNF"
    athlete_meta.loc[finish.str.contains("black", na=False), "group_label"] = "Black Shirt"
    athlete_meta.loc[finish.str.contains("white", na=False), "group_label"] = "White Shirt"
    athlete_meta.loc[(rank <= 10) & (~finish.str.contains("dnf|did not finish|dns", regex=True, na=False)), "group_label"] = "Top 10"

    merged = merged.merge(athlete_meta[["bib", "group_label"]], on="bib", how="left")
    merged["group_label"] = merged["group_label"].fillna("White Shirt")

    # Add athlete labels (name + bib) if available.
    if "name" in df.columns:
        names = (
            df[["bib", "name"]]
            .dropna(subset=["bib"])
            .drop_duplicates(subset=["bib"])
            .copy()
        )
        names["label"] = names.apply(
            lambda r: f"{str(r['name']).strip()} (#{int(float(r['bib']))})"
            if str(r.get("name", "")).strip()
            else f"#{int(float(r['bib']))}",
            axis=1,
        )
        merged = merged.merge(names[["bib", "label"]], on="bib", how="left")
    else:
        merged["label"] = merged["bib"].apply(lambda b: f"#{int(float(b))}")

    fig = go.Figure()
    for bib, g in merged.groupby("bib"):
        g = g.sort_values("x_distance_km")
        label = g["label"].iloc[0] if "label" in g.columns else f"#{int(float(bib))}"
        group_label = g["group_label"].iloc[0] if "group_label" in g.columns else "White Shirt"
        line_color = get_group_color(group_label)
        fig.add_trace(
            go.Scatter(
                x=g["x_distance_km"],
                y=g["delta_h"],
                mode="lines",
                line=dict(width=1.3, color=line_color),
                name=str(label),
                showlegend=False,
                hovertemplate=(
                    f"{label}<br>"
                    f"Group: {group_label}<br>"
                    "Distance: %{x:.1f} km<br>"
                    "Delta vs frontier: %{y:+.2f} h<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=0.0,
        line=dict(color=FRONTIER_COLOR, width=2, dash="dash"),
        annotation_text="Frontier (0 h)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text="Relative To Frontier Time", x=0.01, xanchor="left"),
        xaxis=dict(title="Distance (km)"),
        yaxis=dict(title="Athlete time vs frontier (h)", zeroline=False),
        margin=dict(l=40, r=20, t=45, b=40),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_frontier_pace(selected_year="All", selected_group="All") -> None:
    title_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with title_col:
        st.header("Frontier Pace (P=50% Black Shirt)")
    with info_col:
        with st.popover("ℹ️"):
            st.markdown(
                """
This chart estimates a **50% probability frontier** from model outputs.

- For each split, it interpolates the time at **P(Black Shirt)=0.50**
- Builds a frontier cumulative time and segment speed profile
- Shows how split times accumulate to a total time
                """
            )

    df = _load_model_base()
    df = apply_year_filter(df, selected_year, year_col="year")
    df = apply_group_filter(df, selected_group, finish_col="finish_type", rank_col="split_rank", top10_col="Top10_flag")

    if df.empty:
        st.info("No model data for current filters.")
        return

    cutoff_km = _cutoff_total_km_for_year(selected_year) if selected_year != "All" else float(df["race_distance_km"].max())
    df = df[df["race_distance_km"] <= cutoff_km].copy()
    frontier_df = _frontier_points(df)
    if frontier_df.empty:
        st.info("Could not compute frontier for current filters.")
        return

    course_df = ds_course_profile().copy()
    _render_frontier_chart(frontier_df, course_df, cutoff_km)
    _render_frontier_table(frontier_df)
    st.markdown("#### Frontier on Accumulated Time")
    _render_frontier_on_accumulated(selected_year, selected_group, frontier_df)

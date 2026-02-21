import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_store import df_model as ds_df_model
from utils.race_logic import apply_group_filter, apply_year_filter

PANEL_BG = "#7a7a7a"
SWIM_KM = 3.8
BIKE_KM = 180.0
RUN_KM = 42.2


def _load_filtered_model_data(selected_year, selected_group) -> pd.DataFrame:
    df = ds_df_model().copy()
    if "set" in df.columns:
        df = df[df["set"] == "test"].copy()

    df = apply_year_filter(df, selected_year, year_col="year")
    df = apply_group_filter(
        df,
        selected_group,
        finish_col="finish_type",
        rank_col="split_rank",
        top10_col="Top10_flag",
    )

    for c in ["race_distance_km", "cum_time_seconds", "p_black"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["race_distance_km", "cum_time_seconds", "p_black"])
    df = df[(df["race_distance_km"] >= 0) & (df["cum_time_seconds"] > 0)]
    return df


def _build_surface_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    if df.empty:
        return None, None, None

    work = df.copy()
    work["distance_bin"] = work["race_distance_km"].round(1)
    work["time_h"] = work["cum_time_seconds"] / 3600.0

    time_step_h = 0.25  # 15 min
    work["time_bin_h"] = (work["time_h"] / time_step_h).round() * time_step_h

    x_vals = np.array(sorted(work["distance_bin"].unique()), dtype=float)
    y_min = np.floor(work["time_bin_h"].min() / time_step_h) * time_step_h
    y_max = np.ceil(work["time_bin_h"].max() / time_step_h) * time_step_h
    y_vals = np.arange(y_min, y_max + time_step_h * 0.5, time_step_h, dtype=float)

    z = (
        work.groupby(["time_bin_h", "distance_bin"], as_index=False)["p_black"]
        .mean()
        .pivot(index="time_bin_h", columns="distance_bin", values="p_black")
        .reindex(index=y_vals, columns=x_vals)
    )

    if z.isna().all().all():
        return None, None, None

    z = z.interpolate(axis=0, limit_direction="both")
    z = z.interpolate(axis=1, limit_direction="both")
    z = z.bfill(axis=0).ffill(axis=0).bfill(axis=1).ffill(axis=1)
    z = z.clip(0.0, 1.0)

    return x_vals, y_vals, z.to_numpy(dtype=float)


def _parse_hhmm_to_hours(text: str) -> float | None:
    s = str(text or "").strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) != 2:
            return None
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if h < 0 or m < 0 or m > 59:
        return None
    return h + (m / 60.0)


def _interp_surface_probability(
    x_vals: np.ndarray, y_vals: np.ndarray, z_vals: np.ndarray, x_km: float, y_h: float
) -> float:
    x_clamped = float(np.clip(x_km, float(np.min(x_vals)), float(np.max(x_vals))))
    y_clamped = float(np.clip(y_h, float(np.min(y_vals)), float(np.max(y_vals))))
    z_over_x = np.array([np.interp(x_clamped, x_vals, z_vals[i, :]) for i in range(len(y_vals))], dtype=float)
    z_value = float(np.interp(y_clamped, y_vals, z_over_x))
    return float(np.clip(z_value, 0.0, 1.0))


def _parse_optional_km(text: str, max_km: float) -> float | None:
    s = str(text or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v < 0 or v > max_km:
        return None
    return v


def render_probability_3d(selected_year="All", selected_group="All") -> None:
    title_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with title_col:
        st.header("3D Probability Frontier (Time x Distance x Chance)")
    with info_col:
        with st.popover("ℹ️"):
            st.markdown(
                """
This 3D chart shows the relationship between:
- **Distance (course position)**
- **Cumulative time**
- **Predicted Black-Shirt probability**

Colored surface bands indicate chance zones from low (white) to high (black).
                """
            )

    df = _load_filtered_model_data(selected_year, selected_group)
    if df.empty:
        st.info("No model data available for the current filters.")
        return

    x_vals, y_vals, z_vals = _build_surface_grid(df)
    if x_vals is None:
        st.info("Could not build a stable 3D probability surface for this filter.")
        return

    with st.container():
        st.markdown("##### Check A Race Position In 3D")
        c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
        with c1:
            bike_km_text = st.text_input(
                "Bike km",
                value="",
                placeholder="e.g. 47",
                key="prob3d_bike_km",
            )
        with c2:
            run_km_text = st.text_input(
                "Run km",
                value="",
                placeholder="e.g. 10",
                key="prob3d_run_km",
            )
        with c3:
            time_text = st.text_input("Time (hh:mm)", value="", placeholder="e.g. 10:30", key="prob3d_time_hhmm")
        st.caption("Run-km are mapped as: 3.8 km Swim + 180 km Bike + Run-km.")

    bike_km = _parse_optional_km(bike_km_text, BIKE_KM)
    run_km = _parse_optional_km(run_km_text, RUN_KM)
    has_any_leg_input = bool(str(bike_km_text).strip()) or bool(str(run_km_text).strip())

    if str(bike_km_text).strip() and bike_km is None:
        st.warning(f"Bike km must be between 0 and {BIKE_KM:.1f}.")
    if str(run_km_text).strip() and run_km is None:
        st.warning(f"Run km must be between 0 and {RUN_KM:.1f}.")

    time_h = _parse_hhmm_to_hours(time_text)
    if has_any_leg_input and time_h is None:
        st.warning("Please use time format hh:mm.")

    markers: list[dict] = []
    if time_h is not None:
        if bike_km is not None:
            markers.append(
                {
                    "name": "Bike checkpoint",
                    "symbol": "circle",
                    "distance_total_km": SWIM_KM + float(bike_km),
                }
            )
        if run_km is not None:
            markers.append(
                {
                    "name": "Run checkpoint",
                    "symbol": "x",
                    "distance_total_km": SWIM_KM + BIKE_KM + float(run_km),
                }
            )

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            surfacecolor=z_vals,
            cmin=0.0,
            cmax=1.0,
            colorscale=[
                [0.00, "#ffffff"],
                [0.20, "#f2f2f2"],
                [0.40, "#d9d9d9"],
                [0.50, "#bfbfbf"],
                [0.65, "#8c8c8c"],
                [0.80, "#4d4d4d"],
                [1.00, "#000000"],
            ],
            colorbar=dict(title="P(Black Shirt)", tickformat=".0%"),
            contours={
                "z": dict(show=True, start=0.0, end=1.0, size=0.1, color="rgba(255,255,255,0.45)")
            },
            hovertemplate=(
                "Distance: %{x:.1f} km<br>"
                "Time: %{y:.2f} h<br>"
                "P(Black): %{z:.1%}<extra></extra>"
            ),
            showscale=True,
            opacity=0.98,
        )
    )

    marker_summaries = []
    for m in markers:
        distance_total_km = float(m["distance_total_km"])
        marker_prob = _interp_surface_probability(
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, x_km=distance_total_km, y_h=float(time_h)
        )
        marker_note = ""
        if (
            distance_total_km < float(np.min(x_vals))
            or distance_total_km > float(np.max(x_vals))
            or float(time_h) < float(np.min(y_vals))
            or float(time_h) > float(np.max(y_vals))
        ):
            marker_note = " (point clipped to available data range)"

        fig.add_trace(
            go.Scatter3d(
                x=[distance_total_km],
                y=[float(time_h)],
                z=[marker_prob],
                mode="markers",
                name=str(m["name"]),
                marker=dict(
                    size=9,
                    color="#FF8C00",
                    symbol=str(m["symbol"]),
                    line=dict(color="#111111", width=1),
                ),
                hovertemplate=(
                    f"{m['name']}<br>"
                    "Distance: %{x:.1f} km<br>"
                    "Time: %{y:.2f} h<br>"
                    "P(Black): %{z:.1%}<extra></extra>"
                ),
                showlegend=True,
            )
        )
        marker_summaries.append(
            f"{m['name']}: {distance_total_km:.1f} km @ {time_h:.2f} h -> P(Black) ~ {marker_prob:.1%}{marker_note}"
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=760,
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color="white"),
        scene=dict(
            bgcolor=PANEL_BG,
            xaxis=dict(title="Distance (km)", gridcolor="rgba(220,220,220,0.25)"),
            yaxis=dict(title="Cumulative time (h)", gridcolor="rgba(220,220,220,0.25)"),
            zaxis=dict(
                title="P(Black Shirt)",
                range=[0, 1],
                tickformat=".0%",
                gridcolor="rgba(220,220,220,0.25)",
            ),
            camera=dict(eye=dict(x=1.65, y=1.45, z=0.95)),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    for summary in marker_summaries:
        st.caption(summary)

# accumulated.py

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render(df_long: pd.DataFrame, selected_year, selected_group):
    """
    df_long: Long-DataFrame mit allen Jahren
    selected_year: kommt aus render_header() ("All" oder z.B. 2024)
    selected_group: "All", "Top 10", "Black Shirt", "White Shirt", "DNF"
    """

    if df_long is None or df_long.empty:
        st.info("No data available.")
        return

    # ----------------------------
    # Filter aus dem Header anwenden
    # ----------------------------
    df = df_long.copy()

    if selected_year != "All":
        try:
            year_int = int(selected_year)
            df = df[pd.to_numeric(df["year"], errors="coerce") == year_int]
        except Exception:
            # fallback: wenn selected_year nicht int-konvertierbar ist
            df = df[df["year"].astype(str) == str(selected_year)]


    if selected_group != "All":
        if selected_group == "Top 10" and "overall_rank" in df.columns:
            df = df[df["overall_rank"] <= 10]
        elif selected_group == "Black Shirt" and "finish_type" in df.columns:
            df = df[df["finish_type"] == "Black"]
        elif selected_group == "White Shirt" and "finish_type" in df.columns:
            df = df[df["finish_type"] == "White"]
        elif selected_group == "DNF" and "finish_type" in df.columns:
            df = df[df["finish_type"] == "DNF"]

    if df.empty:
        st.info("No data for the selected filters.")
        return

    # ----------------------------
    # Card Container
    # ----------------------------
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        # --------------------------------------------------
        # Titel + Info-Popover (ℹ️)
        # --------------------------------------------------
        col_title, col_info = st.columns([0.92, 0.08])

        with col_title:
            st.markdown("### Accumulated race progress")

        with col_info:
            with st.popover("ℹ️"):
                st.markdown("""
**What is shown in this chart?**

Each line represents one athlete’s **cumulative race progress** throughout the Norseman course.

- **X-axis**: accumulated distance along the course (km)  
- **Y-axis**: accumulated elapsed race time (hours)  

Athletes are grouped by finish category (Top 10, Black, White, DNF).

**How to interpret this chart**

- A **lower line** indicates a **faster athlete** (less time at the same distance).  
- **Steeper sections** of a line indicate slower pace or more demanding terrain.  
- Lines that **end early** represent **DNF athletes**.

**How this can be used**

- Compare **Top 10 vs. Black / White** to identify where performance gaps emerge.  
- Identify **course sections with high variability**, where lines spread apart.  
- Detect **critical race segments** where many DNFs occur or pacing diverges strongly.
                            

Tip💡 Click and drag a rectangle with your cursor to zoom in the graph. This could be interesting at cut-off points E.g. 216 km (2018, 2019, 2021, 2022) and 221 km (2024, 2025)
                """)

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig = _create_accumulated_figure(df)
        st.plotly_chart(fig, width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)


def _create_accumulated_figure(df: pd.DataFrame) -> go.Figure:
    df = df.copy()

    # ----------------------------
    # Athleten-ID bestimmen
    # ----------------------------
    athlete_col = next(
        (c for c in ["athlete_id", "bib", "participant_id", "bib_number"] if c in df.columns),
        None,
    )

    if athlete_col is None and {"year", "bib"}.issubset(df.columns):
        df["athlete_id"] = df["year"].astype(str) + "_" + df["bib"].astype(str)
        athlete_col = "athlete_id"

    if athlete_col is None:
        df["athlete_id"] = df.index
        athlete_col = "athlete_id"

    # ----------------------------
    # Distanz
    # ----------------------------
    if "race_distance_km" in df.columns:
        df["cum_distance_km"] = pd.to_numeric(df["race_distance_km"], errors="coerce")
    elif "split_distance_km" in df.columns:
        df["cum_distance_km"] = pd.to_numeric(df["split_distance_km"], errors="coerce")
    else:
        df["cum_distance_km"] = 0.0

    df["cum_distance_km"] = df["cum_distance_km"].clip(0, 226)

    # ----------------------------
    # Leg-relative X-Achse (Swim 0-3.8, Bike 0-180, Run 0-42.2)
    # ----------------------------
    SWIM_KM = 3.8
    BIKE_KM = 180.0
    RUN_KM = 42.2
    GAP_KM = 6.0  

    swim_end = SWIM_KM
    bike_end = SWIM_KM + BIKE_KM

    def _leg_from_cum(x):
        if pd.isna(x):
            return "unknown"
        if x <= swim_end + 1e-9:
            return "swim"
        if x <= bike_end + 1e-9:
            return "bike"
        return "run"

    df["leg"] = df["cum_distance_km"].apply(_leg_from_cum)

    df["leg_km"] = df["cum_distance_km"]
    df.loc[df["leg"] == "bike", "leg_km"] = df.loc[df["leg"] == "bike", "cum_distance_km"] - swim_end
    df.loc[df["leg"] == "run", "leg_km"] = df.loc[df["leg"] == "run", "cum_distance_km"] - bike_end

    swim_offset = 0.0
    bike_offset = SWIM_KM + GAP_KM
    run_offset = (SWIM_KM + GAP_KM) + BIKE_KM + GAP_KM

    df["x_plot"] = df["leg_km"]
    df.loc[df["leg"] == "bike", "x_plot"] = df.loc[df["leg"] == "bike", "leg_km"] + bike_offset
    df.loc[df["leg"] == "run", "x_plot"] = df.loc[df["leg"] == "run", "leg_km"] + run_offset

    x_max = run_offset + RUN_KM

    # ----------------------------
    # Zeit
    # ----------------------------
    df["cum_time_hours"] = df["cum_time_seconds"] / 3600
    df = df.sort_values([athlete_col, "cum_distance_km"])
    df["cum_time_hours"] = df.groupby(athlete_col)["cum_time_hours"].cummax()

    # ----------------------------
    # Kategorien
    # ----------------------------
    base_cat_col = next(
        (c for c in ["category", "finish_group", "finish_type_group", "finish_type"] if c in df.columns),
        None,
    )
    df["category"] = df[base_cat_col].astype(str) if base_cat_col else "Other"

    if "overall_rank" in df.columns:
        df.loc[df["overall_rank"] <= 10, "category"] = "Top10"

    color_map = {
        "DNF": "#FF4B4B",
        "White": "#FFFFFF",
        "Black": "#000000",
        "Top10": "#2ECC71",
    }

    fig = go.Figure()

    # ----------------------------
    # Linien
    # ----------------------------
    for category, df_cat in df.groupby("category"):
        first = True

        for _, df_ath in df_cat.groupby(athlete_col):
            base_opacity = (
                1.0 if category == "Top10"
                else 0.75 if category == "DNF"
                else 0.70 if category == "Black"
                else 0.65 if category == "White"
                else 0.60
            )

            fig.add_trace(
                go.Scatter(
                    x=df_ath["x_plot"],
                    y=df_ath["cum_time_hours"],
                    mode="lines",
                    line=dict(
                        color=color_map.get(category, "#9b9b9b"),
                        width=1.8 if category == "Top10" else (1.6 if category == "DNF" else 1.0),
                    ),
                    opacity=1.0 if first else base_opacity,
                    name=category,
                    legendgroup=category,
                    showlegend=first,
                    hoverinfo="none",
                )
            )
            first = False

    # ----------------------------
    # Split-Linien + Baseline
    # ----------------------------
    GRID_COLOR = "#C8C8C8"
    split_positions = sorted(df["x_plot"].dropna().round(1).unique())

    shapes = [
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=p,
            x1=p,
            y0=0,
            y1=1,
            line=dict(color=GRID_COLOR, width=1, dash="dot"),
            layer="below",
        )
        for p in split_positions
    ]

    # Leg-Trenner (deutlich)
    for p in [SWIM_KM, bike_offset + BIKE_KM]:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=p,
                x1=p,
                y0=0,
                y1=1,
                line=dict(color="white", width=2),
                layer="above",
            )
        )

    shapes.append(
        dict(
            type="line",
            xref="paper",
            yref="paper",
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            line=dict(color="white", width=2),
            layer="above",
        )
    )

    # ----------------------------
    # Layout
    # ----------------------------
    tickvals = [
    0.0,
    SWIM_KM,
    bike_offset + BIKE_KM,
    run_offset + RUN_KM,
]

    ticktext = [
        "0",
        f"{SWIM_KM:g}",
        f"{BIKE_KM:g}",
        f"{RUN_KM:g}",
]


    fig.update_layout(
        plot_bgcolor="#7A7A7A",
        paper_bgcolor="#7A7A7A",
        font=dict(color="white"),
        legend=dict(
            bgcolor="#7A7A7A",
            font=dict(color="white", size=30),
            itemsizing="constant",
            itemwidth=40,
        ),
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="closest",
        shapes=shapes,
        xaxis=dict(
            title=dict(text="Distance (km)", font=dict(color="white", size=25)),
            range=[0, x_max],
            showgrid=True,
            gridcolor=GRID_COLOR,
            griddash="dot",
            zeroline=False,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(color="white", size=14),
        ),
        yaxis=dict(
            title=dict(text="Time (hours)", font=dict(color="white", size=25)),
            showgrid=True,
            gridcolor=GRID_COLOR,
            griddash="dot",
            zeroline=False,
            tickfont=dict(color="white", size=14),
        ),
        annotations=[
            dict(
                x=SWIM_KM / 2,
                y=1.06,
                xref="x",
                yref="paper",
                text="Swim",
                showarrow=False,
                font=dict(color="white", size=16),
            ),
            dict(
                x=bike_offset + BIKE_KM / 2,
                y=1.06,
                xref="x",
                yref="paper",
                text="Bike",
                showarrow=False,
                font=dict(color="white", size=16),
            ),
            dict(
                x=run_offset + RUN_KM / 2,
                y=1.06,
                xref="x",
                yref="paper",
                text="Run",
                showarrow=False,
                font=dict(color="white", size=16),
            ),
        ],
    )

    return fig

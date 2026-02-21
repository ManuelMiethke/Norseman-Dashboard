import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from utils.race_logic import get_group_color

# --------------------------------------------------
# Pfad zur XGBoost-Prognose-Datei
# --------------------------------------------------
@st.cache_data
def load_black_prob_data() -> pd.DataFrame:
    # Bridge: Model CSV kommt zentral aus data_store
    import data_store
    df = data_store.df_model().copy()

    df = df.rename(columns={
        "race_distance_km": "distance_km",
        "p_black": "black_prob",
        "finish_type": "finish_raw",
    })

    if "set" in df.columns:
        df = df[df["set"] == "test"].copy()

    def map_finish_group(s: str) -> str:
        s_low = str(s).strip().lower()
        if "black" in s_low:
            return "Black Shirt"
        if "white" in s_low:
            return "White Shirt"
        return "Other"

    df["finish_group"] = df["finish_raw"].apply(map_finish_group)
    df = df[df["finish_group"].isin(["Black Shirt", "White Shirt"])].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["bib"] = pd.to_numeric(df["bib"], errors="coerce")
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["black_prob"] = pd.to_numeric(df["black_prob"], errors="coerce")

    df = df.dropna(subset=["year", "bib", "distance_km", "black_prob"])
    return df



def render_blackshirt_probability(selected_year="All", selected_group="All"):
    df = load_black_prob_data()

    col_title, col_info = st.columns([0.92, 0.08])
    with col_title:
        st.subheader("Black Shirt Probability over the Norseman Course")
    with col_info:
        with st.popover("ℹ️"):
            st.markdown("""
    **What is shown in this chart?**

    This visualization shows how the **predicted probability of achieving a Black Shirt**
    evolves throughout the Norseman course.

    - **X-axis**: distance along the course, separated by Swim, Bike and Run  
    - **Y-axis**: probability of finishing with a **Black Shirt**

    - **Thin lines**: individual athletes  
    - **Dashed line**: median probability at each point  
    - **Dotted line**: mean probability at each point  

    **How to interpret this chart**

    - Rising curves indicate **improving chances** of securing a Black Shirt  
    - Falling curves indicate **increasing risk** of missing the cut-off  
    - Strong changes often highlight **decisive race sections**

    **How this can be used**

    - Identify **critical segments** that matter most for Black Shirt success  
    - Compare how probability trajectories differ between groups  
    - Understand **when the race outcome becomes predictable**
        """)

    # DNF soll nichts anzeigen
    if selected_group == "DNF":
        st.info("For the Group DNF, the Black-Shirt-Probability will not be shown.")
        return

    # Jahr(e) filtern
    if selected_year != "All":
        df = df[df["year"] == int(selected_year)]

    if df.empty:
        st.warning("No Data.")
        return

    # X-Achsen-Range
    SWIM_LEN = 3.8
    BIKE_LEN = 180.0
    RUN_LEN_DISPLAY = 42.2
    LEG_GAP = 1.5

    swim_end = SWIM_LEN
    bike_end = SWIM_LEN + BIKE_LEN

    # Leg-Info + X-Achse mit Gaps je Leg
    df = df.copy()
    df["leg"] = "Run"
    df.loc[df["distance_km"] <= swim_end, "leg"] = "Swim"
    df.loc[(df["distance_km"] > swim_end) & (df["distance_km"] <= bike_end), "leg"] = "Bike"

    df["leg_km"] = df["distance_km"]
    df.loc[df["leg"] == "Bike", "leg_km"] = df.loc[df["leg"] == "Bike", "distance_km"] - swim_end
    df.loc[df["leg"] == "Run", "leg_km"] = df.loc[df["leg"] == "Run", "distance_km"] - bike_end

    leg_offset = {
        "Swim": 0.0,
        "Bike": swim_end + LEG_GAP,
        "Run": swim_end + LEG_GAP + BIKE_LEN + LEG_GAP,
    }
    df["x_plot"] = df["leg_km"] + df["leg"].map(leg_offset)

    run_leg_max = df.loc[df["leg"] == "Run", "leg_km"].max()
    run_leg_max = float(run_leg_max) if pd.notna(run_leg_max) else 0.0
    run_display = max(RUN_LEN_DISPLAY, run_leg_max)
    axis_end = leg_offset["Run"] + run_display

    # Top-10-Bibs bestimmen (letzter Split pro Athlet, pro Jahr falls All gewählt)
    group_keys = ["year", "bib"] if selected_year == "All" else ["bib"]
    df_sorted = df.sort_values(group_keys + ["distance_km"])
    last_split = df_sorted.groupby(group_keys).tail(1)
    top10_keys = last_split[last_split["split_rank"] <= 10][group_keys]
    df = df.merge(top10_keys.assign(is_top10=True), on=group_keys, how="left")
    df["is_top10"] = (df["is_top10"].fillna(False).infer_objects(copy=False))
    # ============================
    # Gruppen-Logik
    # ============================
    groups_data = []

    if selected_group == "All":
        df_top10 = df[df["is_top10"]].copy()
        df_black_rest = df[(df["finish_group"] == "Black Shirt") & (~df["is_top10"])].copy()
        df_white = df[df["finish_group"] == "White Shirt"].copy()

        groups_data = [
            ("Top 10", df_top10),
            ("Black Shirt", df_black_rest),
            ("White Shirt", df_white),
        ]

    elif selected_group == "Top 10":
        df_top10 = df[df["is_top10"]].copy()
        groups_data = [("Top 10", df_top10)]

    elif selected_group == "Black Shirt":
        df_black = df[df["finish_group"] == "Black Shirt"].copy()
        groups_data = [("Black Shirt", df_black)]

    elif selected_group == "White Shirt":
        df_white = df[df["finish_group"] == "White Shirt"].copy()
        groups_data = [("White Shirt", df_white)]

    else:
        st.warning("Unbekannte Gruppe im Filter.")
        return

    if all(gdf.empty for _, gdf in groups_data):
        st.warning("Keine Daten nach Anwendung der Filter vorhanden.")
        return

    # ============================
    # Subplots anlegen
    # ============================
    n_cols = len(groups_data)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[g[0] for g in groups_data],
        horizontal_spacing=0.03 if n_cols > 1 else 0.01,
    )

    color_map = {
        "Top 10": get_group_color("Top 10", scheme="blackshirt_prob"),
        "Black Shirt": get_group_color("Black Shirt", scheme="blackshirt_prob"),
        "White Shirt": get_group_color("White Shirt", scheme="blackshirt_prob"),
    }

    MEDIAN_COLOR = "#FF8C00"
    MEAN_COLOR = "blue"

    for col_idx, (grp_name, grp_df) in enumerate(groups_data, start=1):
        if grp_df.empty:
            continue

        first_for_legend = True
        for bib, bib_df in grp_df.groupby("bib"):
            bib_df = bib_df.sort_values("distance_km")

            fig.add_trace(
                go.Scatter(
                    x=bib_df["x_plot"],
                    y=bib_df["black_prob"],
                    mode="lines",
                    line=dict(color=color_map[grp_name], width=1.2),
                    opacity=0.35,
                    name=grp_name,
                    showlegend=first_for_legend,
                    hovertemplate=(
                        f"Group: {grp_name}<br>"
                        "Bib: %{customdata[0]}<br>"
                        "Leg: %{customdata[1]}<br>"
                        "Leg Distance: %{customdata[2]:.1f} km<br>"
                        "Total Distance: %{customdata[3]:.1f} km<br>"
                        "P(Black Shirt): %{y:.2f}<extra></extra>"
                    ),
                    customdata=bib_df[["bib", "leg", "leg_km", "distance_km"]],
                ),
                row=1,
                col=col_idx,
            )
            first_for_legend = False

        stats_df = (
            grp_df.groupby("x_plot")["black_prob"]
            .agg(median="median", mean="mean")
            .reset_index()
            .sort_values("x_plot")
        )

        fig.add_trace(
            go.Scatter(
                x=stats_df["x_plot"],
                y=stats_df["median"],
                mode="lines",
                line=dict(color=MEDIAN_COLOR, width=3, dash="dash"),
                opacity=0.95,
                name="Median",
                showlegend=(col_idx == 1),
                hovertemplate="Median<br>Position: %{x:.1f} km<br>P(Black Shirt): %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.add_trace(
            go.Scatter(
                x=stats_df["x_plot"],
                y=stats_df["mean"],
                mode="lines",
                line=dict(color=MEAN_COLOR, width=2, dash="dot"),
                opacity=0.85,
                name="Mean",
                showlegend=(col_idx == 1),
                hovertemplate="Mean<br>Position: %{x:.1f} km<br>P(Black Shirt): %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

    # ============================
    # Split-Station Linien 
    # ============================
    # Swim: nur 3.8 (Swim-Ende)
    swim_splits = [3.8]

    # Bike: offizielle Stationen (Leg-KM)
    bike_splits = [6, 11, 16, 20, 28, 36, 47, 66, 90, 94, 101, 113, 123, 134, 142, 152, 179, 180]

    # Run: offizielle Stationen (Leg-KM)
    run_splits = [5, 10, 15, 20, 25, 32.5, 33, 37.5, 40, 42.2]

    split_x_marks = []
    # Swim
    for km in swim_splits:
        split_x_marks.append(km)  # Swim Offset = 0
    # Bike
    for km in bike_splits:
        split_x_marks.append(leg_offset["Bike"] + km)
    # Run
    for km in run_splits:
        split_x_marks.append(leg_offset["Run"] + km)

    # nur die, die in deiner Achse sichtbar sind
    split_x_marks = [x for x in split_x_marks if 0 <= x <= axis_end]

    # exakt der Stil wie dein Bike-180 Beispiel
    VLINE_STYLE = dict(width=1, dash="dot", color="#b0b0b0")

    for x_mark in split_x_marks:
        for c in range(1, n_cols + 1):
            xref = "x" if c == 1 else f"x{c}"
            yref = "y" if c == 1 else f"y{c}"
            fig.add_shape(
                type="line",
                x0=x_mark, x1=x_mark,
                y0=0, y1=1,
                xref=xref,
                yref=yref,
                line=VLINE_STYLE,
                opacity=0.6,
                layer="above",
            )

    tickvals = [
        swim_end,  # Swim 3.8
        leg_offset["Bike"] + 36,  # Bike 36
        leg_offset["Bike"] + 90,  # Bike 90
        leg_offset["Bike"] + BIKE_LEN,  # Bike 180
        leg_offset["Run"] + 21,  # Run 21
        leg_offset["Run"] + run_display,  # Run end (42.2 typ.)
    ]
    ticktext = [
        "Swim 3.8",
        "Bike 36",
        "Bike 90",
        "Bike 180",
        "Run 21",
        f"Run {run_display:.1f}",
    ]

    fig.update_xaxes(
        range=[0, axis_end],
        title_text="Leg Distance (km)",
        title_font=dict(size=22),
        tickfont=dict(size=14),
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    fig.update_yaxes(range=[0, 1], title_text="P(Black Shirt)", title_font=dict(size=22), tickfont=dict(size=14))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#7A7A7A",
        plot_bgcolor="#7A7A7A",
        height=650,
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(font=dict(size=22)),
    )

    st.plotly_chart(fig, width="stretch")

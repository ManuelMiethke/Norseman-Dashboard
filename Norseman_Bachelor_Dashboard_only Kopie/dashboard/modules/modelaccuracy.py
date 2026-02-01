import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from plotly.subplots import make_subplots

# Bridge data layer 
try:
    import data_store
except Exception:
    data_store = None

# --------------------------------------------------
# Header mit Year-/Group-Filter
# --------------------------------------------------
try:
    from header import render_header
except ImportError:
    render_header = None

# --------------------------------------------------
# Pfade
# --------------------------------------------------
DATA_PATH = None  # provided via data_store.df_model() 
COURSE_PROFILE_FILE = None  # provided via data_store.course_profile() 

# Kursprofil-Spalten
COL_COURSE_DIST = "distance_km"
COL_ELEV_NORSE = "elev_norseman_m"


# --------------------------------------------------
# Data Loading
# --------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [
        "year", "bib", "name", "finish_type", "split_key", "leg",
        "race_distance_km", "cum_time_seconds", "segment_speed_kmh",
        "split_rank", "split_rank_relative", "y_true", "p_black",
        "y_pred", "is_error", "error_type", "set",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if "set" in df.columns:
        df = df[df["set"] == "test"].copy()

    df["race_distance_km"] = pd.to_numeric(df["race_distance_km"], errors="coerce")
    df = df.dropna(subset=["race_distance_km"])
    return df


@st.cache_data
def load_course_profile(path: Path) -> pd.DataFrame:
    course = pd.read_csv(path)

    course[COL_COURSE_DIST] = pd.to_numeric(course[COL_COURSE_DIST], errors="coerce")
    course[COL_ELEV_NORSE] = pd.to_numeric(course[COL_ELEV_NORSE], errors="coerce")

    course = course.dropna(subset=[COL_COURSE_DIST, COL_ELEV_NORSE])
    return course.sort_values(COL_COURSE_DIST)


# --------------------------------------------------
# Filtering
# --------------------------------------------------
def apply_filters(df: pd.DataFrame, year, group) -> pd.DataFrame:
    if year not in (None, "All"):
        df = df[df["year"] == int(year)]

    if group not in (None, "All"):
        if group == "Top 10":
            df = df[df["split_rank"] <= 10]
        elif group == "Black Shirt":
            df = df[df["finish_type"].str.contains("Black", case=False, na=False)]
        elif group == "White Shirt":
            df = df[df["finish_type"].str.contains("White", case=False, na=False)]
        elif group == "DNF":
            df = df[df["finish_type"].str.contains("DNF", case=False, na=False)]

    return df


# --------------------------------------------------
# Accuracy aggregation
# --------------------------------------------------
def compute_accuracy_by_station(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["correct"] = df["y_true"] == df["y_pred"]

    agg = (
        df.groupby(["split_key", "race_distance_km"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_obs=("correct", "size"),
        )
        .sort_values("race_distance_km")
    )

    agg["accuracy_pct"] = agg["accuracy"] * 100.0
    return agg


# --------------------------------------------------
# Helper: X-Labels ausdünnen
# --------------------------------------------------
def make_sparse_tick_labels(acc_df: pd.DataFrame, min_spacing_km: float = 6.0):
    xs = acc_df["race_distance_km"].astype(float).tolist()
    labels = acc_df["split_key"].astype(str).tolist()

    ticktext = [""] * len(xs)
    last_labeled_x = None

    for i, x in enumerate(xs):
        if last_labeled_x is None or (x - last_labeled_x) >= min_spacing_km:
            ticktext[i] = labels[i]
            last_labeled_x = x

    if xs:
        ticktext[-1] = labels[-1]

    return xs, ticktext


# --------------------------------------------------
# NEW: Elevation Fläche gradient-colored (Steigung)
# --------------------------------------------------
def _color_for_slope_pct(p: float) -> str:
    """
    Gleiche Logik/Farben wie beim Band, nur als Flächen-Fill.
    """
    if p <= -4:
        return "rgba(0, 200, 255, 0.60)"   # strong downhill
    if p <= -1:
        return "rgba(0, 160, 255, 0.50)"   # mild downhill
    if p < 1:
        return "rgba(255, 255, 255, 0.14)" # flat-ish
    if p < 4:
        return "rgba(255, 80, 200, 0.55)"  # mild uphill
    return "rgba(180, 0, 255, 0.60)"       # steep uphill


def add_gradient_colored_elevation(
    fig: go.Figure,
    course_df: pd.DataFrame,
    *,
    sample_every_km: float = 0.5,
):
    """
    Färbt die Elevation-Fläche segmentweise nach Steigung ein.

    Performance:
    - Kursprofil wird gebinnt (sample_every_km)
    - Segmente mit gleicher Farbe werden zu Blöcken zusammengefasst,
      damit nicht hunderte Traces entstehen.
    """
    if course_df.empty:
        return

    df = course_df[[COL_COURSE_DIST, COL_ELEV_NORSE]].dropna().sort_values(COL_COURSE_DIST).copy()
    df = df.drop_duplicates(subset=[COL_COURSE_DIST])

    # Ausdünnen (binning)
    if sample_every_km and sample_every_km > 0:
        df["_bin"] = (df[COL_COURSE_DIST] / sample_every_km).round().astype(int)
        df = df.groupby("_bin", as_index=False).agg(
            **{
                COL_COURSE_DIST: (COL_COURSE_DIST, "mean"),
                COL_ELEV_NORSE: (COL_ELEV_NORSE, "mean"),
            }
        )

    xs = df[COL_COURSE_DIST].astype(float).to_numpy()
    ys = df[COL_ELEV_NORSE].astype(float).to_numpy()
    if len(xs) < 2:
        return

    dx_km = xs[1:] - xs[:-1]
    dy_m = ys[1:] - ys[:-1]
    dx_m = dx_km * 1000.0
    slope_pct = (dy_m / dx_m) * 100.0
    slope_pct = pd.Series(slope_pct).fillna(0).to_numpy()

    colors = [_color_for_slope_pct(float(p)) for p in slope_pct]

    # Blöcke gleicher Farbe zusammenfassen
    blocks = []
    start_idx = 0
    for i in range(1, len(colors)):
        if colors[i] != colors[i - 1]:
            blocks.append((start_idx, i, colors[i - 1]))
            start_idx = i
    blocks.append((start_idx, len(colors), colors[-1]))  # letzter Block

    # Flächenblöcke zeichnen (ohne Legende)
    for (a, b, col) in blocks:
        # Segment-Range: xs[a] ... xs[b]
        x_seg = xs[a:(b + 1)]
        y_seg = ys[a:(b + 1)]

        fig.add_trace(
            go.Scatter(
                x=x_seg,
                y=y_seg,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor=col,
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=True,
        )
        fig.data[-1].update(zorder=1)

    # Outline (eine Linie) für Legende + Klarheit
    fig.add_trace(
        go.Scatter(
            x=course_df[COL_COURSE_DIST],
            y=course_df[COL_ELEV_NORSE],
            name="Norseman elevation",
            mode="lines",
            line=dict(width=1.5, color="#4C5FD7"),
            hovertemplate="km %{x:.1f}<br>Elev: %{y:.0f} m<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.data[-1].update(zorder=2)


# --------------------------------------------------
# Plot
# --------------------------------------------------
def build_figure(acc_df: pd.DataFrame, course_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --------------------------------------------------
    # Elevation Fläche: gradient-colored (ersetzt die alte einfarbige Fläche)
    # --------------------------------------------------
    add_gradient_colored_elevation(
        fig,
        course_df,
        sample_every_km=0.5,  # 0.25 feiner / 1.0 grober
    )

    # --------------------------------------------------
    # Accuracy (FOREGROUND)
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=acc_df["race_distance_km"],
            y=acc_df["accuracy_pct"],
            mode="lines+markers",
            line=dict(width=4, color="#FFB000"),
            marker=dict(size=8, color="#FFB000"),
            name="Model accuracy",
            customdata=acc_df[["split_key", "n_obs"]].values,
            hovertemplate=(
                "Station: %{customdata[0]}<br>"
                "Distance: %{x:.1f} km<br>"
                "Accuracy: %{y:.2f}%<br>"
                "n = %{customdata[1]}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.data[-1].update(zorder=10)

    # --------------------------------------------------
    # Vertikale Split-Linien
    # --------------------------------------------------
    for x in acc_df["race_distance_km"].astype(float):
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1),
            layer="below",
        )

    x_max = max(
        float(acc_df["race_distance_km"].max()),
        float(course_df[COL_COURSE_DIST].max()),
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        template="plotly_dark",
        height=740,
        paper_bgcolor="#7A7A7A",
        plot_bgcolor="#7A7A7A",
        margin=dict(l=80, r=110, t=30, b=90),
        hovermode="x unified",
        legend=dict(
            bgcolor="#7A7A7A",
            font=dict(color="white", size=30),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            traceorder="reversed",  # Model accuracy oben
        ),
    )

    # --------------------------------------------------
    # X-Axis (ausgedünnt)
    # --------------------------------------------------
    tickvals, ticktext = make_sparse_tick_labels(acc_df, min_spacing_km=6.0)

    fig.update_xaxes(
        title_text="Distance (km)",
        title_font=dict(size=25),
        tickfont=dict(size=20),
        range=[0, x_max],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-60,
        ticklabelposition="outside",
        automargin=True,
        showgrid=False,
        zeroline=False,
    )

    # --------------------------------------------------
    # Y-Axes
    # --------------------------------------------------
    fig.update_yaxes(
        title_text="Model accuracy (%)",
        title_font=dict(size=25),
        tickfont=dict(size=20),
        secondary_y=False,
        range=[
            max(0, acc_df["accuracy_pct"].min() - 3),
            min(100, acc_df["accuracy_pct"].max() + 3),
        ],
        tickformat=".0f",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.2)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Elevation (m)",
        title_font=dict(size=25),
        tickfont=dict(size=20),
        secondary_y=True,
        rangemode="tozero",
        showgrid=False,
        zeroline=False,
    )

    return fig


# --------------------------------------------------
# Streamlit Wrapper
# --------------------------------------------------
def render_model_accuracy(selected_year="All", selected_group="All"):
    # --------------------------------------------------
    # Bridge load 
    # --------------------------------------------------
    import data_store
    df = data_store.df_model()
    course_df = data_store.course_profile()

    df_filtered = apply_filters(df, selected_year, selected_group)
    if df_filtered.empty:
        st.info("No data available for the selected filters.")
        return

    acc_df = compute_accuracy_by_station(df_filtered)

    year_txt = "all years" if selected_year == "All" else f"year {selected_year}"
    group_txt = "all athletes" if selected_group == "All" else f"group: {selected_group}"

    col_title, col_info = st.columns([0.92, 0.08])
    with col_title:
        st.subheader(f"Model accuracy by timing station ({year_txt}, {group_txt})")
    with col_info:
        with st.popover("ℹ️"):
            st.markdown("""
**What is shown in this chart?**

This visualization shows how accurately the prediction model classifies athletes
(*Black / White finish*) at each official timing station along the Norseman course.

- **Amber line**: Model accuracy (%) at each timing station  
- **Colored elevation area**: Course profile with slope-based coloring  

**How to interpret this chart**

- Rising accuracy indicates that outcomes become more predictable as the race progresses  
- Drops in accuracy often coincide with variable or technically demanding sections  

**How this can be used**

- Identify course sections where prediction is most uncertain  
- Compare early vs. late-race predictability for strategic insights
            """)

    fig = build_figure(acc_df, course_df)
    st.plotly_chart(fig, width="stretch")


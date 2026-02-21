import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.race_logic import get_group_color, is_critical_40_group

try:
    import data_store
except Exception:
    data_store = None


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_long_data() -> pd.DataFrame:
    if data_store is None:
        raise RuntimeError("data_store not available. Place data_store.py next to main.py.")

    df = data_store.df_long()

    df["split_rank"] = pd.to_numeric(df.get("split_rank"), errors="coerce")
    df["race_distance_km"] = pd.to_numeric(df.get("race_distance_km"), errors="coerce")
    df["bib"] = pd.to_numeric(df.get("bib"), errors="coerce")
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")

    df["finish_type"] = df["finish_type"].astype(str) if "finish_type" in df.columns else ""
    df["name"] = df["name"].astype(str) if "name" in df.columns else ""

    df = df.dropna(subset=["bib", "year", "race_distance_km", "split_rank"])
    df["bib"] = df["bib"].astype(int)
    df["split_rank"] = df["split_rank"].astype(int)

    return df


# --------------------------------------------------
# Race structure (year-dependent)
# --------------------------------------------------
def race_structure_for_year(year: int) -> dict:
    SWIM_KM = 3.8
    BIKE_KM = 180.0

    if year in {2018, 2019, 2021, 2022}:
        RUN_KM = 32.5
    else:
        RUN_KM = 37.5

    swim_end = SWIM_KM
    bike_end = SWIM_KM + BIKE_KM
    run_end = SWIM_KM + BIKE_KM + RUN_KM

    # extra labeled split markers (absolute race distance, not leg-relative)
    x_bike_47 = swim_end + 47.0
    x_bike_90 = swim_end + 90.0
    x_bike_142 = swim_end + 142.0
    x_run_20 = bike_end + 20.0

    return {
        "swim_end": swim_end,
        "bike_end": bike_end,
        "run_end": run_end,
        "run_km": RUN_KM,
        "extra_ticks": [
            (x_bike_47, "Bike\n47"),
            (x_bike_90, "Bike\n90"),
            (x_bike_142, "Bike\n142"),
            (x_run_20, "Run\n20"),
        ],
    }


# --------------------------------------------------
# Color / group logic
# --------------------------------------------------
def is_dnf(finish_type: str) -> bool:
    t = (finish_type or "").lower()
    return ("dnf" in t) or ("did not finish" in t) or ("dns" in t)


def is_black_shirt(finish_type: str) -> bool:
    return "black" in (finish_type or "").lower()


def is_white_shirt(finish_type: str) -> bool:
    return "white" in (finish_type or "").lower()


def athlete_color(finish_type: str, top10_at_cutoff: bool) -> str:
    # Priority: DNF > Top10 > Black > White > fallback
    if is_dnf(finish_type):
        return get_group_color("DNF", scheme="rank_progression")
    if top10_at_cutoff:
        return get_group_color("Top 10", scheme="rank_progression")
    if is_black_shirt(finish_type):
        return get_group_color("Black Shirt", scheme="rank_progression")
    if is_white_shirt(finish_type):
        return get_group_color("White Shirt", scheme="rank_progression")
    return get_group_color("White Shirt", scheme="rank_progression")


def _get_group_from_session_state() -> str:
    for k in ("selected_group", "group_filter", "filter_group", "group"):
        if k in st.session_state and st.session_state[k]:
            return str(st.session_state[k]).strip()
    return "All"


def athlete_in_group(row: pd.Series, group: str) -> bool:
    group = (group or "All").strip()
    if group == "All":
        return True
    if group == "Top 10":
        return bool(row.get("top10_at_cutoff", False))
    if group == "Black Shirt":
        return is_black_shirt(row.get("finish_type", ""))
    if group == "White Shirt":
        return is_white_shirt(row.get("finish_type", ""))
    if group == "DNF":
        return is_dnf(row.get("finish_type", ""))
    if is_critical_40_group(group):
        try:
            rank = float(row.get("last_rank_at_cutoff"))
            return 140 <= rank <= 180
        except Exception:
            return False
    return True


# --------------------------------------------------
# Figure
# --------------------------------------------------
def create_rank_progression_figure(
    df_year: pd.DataFrame,
    year_int: int,
    group: str,
) -> go.Figure:
    structure = race_structure_for_year(year_int)
    cutoff_km = structure["run_end"]
    cutoff_tol_km = 0.2

    df = df_year[df_year["race_distance_km"] <= cutoff_km].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#7A7A7A",
            plot_bgcolor="#7A7A7A",
            font=dict(color="#FFFFFF"),
            showlegend=False,
            height=650,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        fig.add_annotation(
            text="No data available for this year/cutoff.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#FFFFFF"),
        )
        return fig

    # Keep only athletes that actually reached the cut-off distance.
    max_dist_per_bib = (
        df_year.groupby("bib", as_index=False)["race_distance_km"]
        .max()
        .rename(columns={"race_distance_km": "max_race_distance_km"})
    )
    reached_cutoff_bibs = set(
        max_dist_per_bib.loc[
            max_dist_per_bib["max_race_distance_km"] >= (cutoff_km - cutoff_tol_km), "bib"
        ].astype(int)
    )

    df = df[df["bib"].isin(reached_cutoff_bibs)].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#7A7A7A",
            plot_bgcolor="#7A7A7A",
            font=dict(color="#FFFFFF"),
            showlegend=False,
            height=650,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        fig.add_annotation(
            text="No athletes reached the cut-off for this selection.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#FFFFFF"),
        )
        return fig

    # last rank at cutoff per athlete
    last_rows = (
        df.sort_values(["bib", "race_distance_km"])
          .groupby("bib", as_index=False)
          .tail(1)
          .loc[:, ["bib", "split_rank"]]
          .rename(columns={"split_rank": "last_rank_at_cutoff"})
    )

    meta = (
        df[["bib", "name", "finish_type"]]
        .drop_duplicates(subset=["bib"])
        .copy()
    )

    summary = meta.merge(last_rows, on="bib", how="left")
    summary["top10_at_cutoff"] = summary["last_rank_at_cutoff"].fillna(999999).astype(int).le(10)
    summary["color"] = summary.apply(
        lambda r: athlete_color(r.get("finish_type", ""), bool(r["top10_at_cutoff"])),
        axis=1
    )

    summary = summary[summary.apply(lambda r: athlete_in_group(r, group), axis=1)].copy()

    summary = summary.sort_values("last_rank_at_cutoff", na_position="last")

    fig = go.Figure()

    for _, r in summary.iterrows():
        bib = int(r["bib"])
        name = (r.get("name") or "").strip()
        label = f"{name} (#{bib})" if name else f"#{bib}"

        df_a = df[df["bib"] == bib].sort_values("race_distance_km")
        if df_a.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_a["race_distance_km"],
                y=df_a["split_rank"],
                mode="lines",
                name=label,
                line=dict(width=1.4, color=r["color"]),
                showlegend=False,
                hovertemplate=f"{label}<br>km %{{x:.1f}}<br>rank %{{y}}<extra></extra>",
            )
        )

    # ---------------------------
    # Y axis fix (rank must start at 1)
    # ---------------------------
    max_rank = int(df["split_rank"].max())
    fig.update_yaxes(
        title=dict(text="Rank", font=dict(size=25)),
        autorange=False,              
        range=[max_rank + 2, 1],      
        tickmode="linear",
        tick0=1,                     
        dtick=10,
        gridcolor="#555555",
        zeroline=False,
    )

    # ---------------------------
    # X axis with extra split ticks
    # ---------------------------
    base_ticks = [
        (0.0, "Start"),
        (structure["swim_end"], "Swim 3.8 km"),
        *structure["extra_ticks"],  
        (structure["bike_end"], "Bike 180 km"),
        (structure["run_end"], f"Run {structure['run_km']} km"),
    ]

    seen = set()
    tick_vals, tick_text = [], []
    for x, t in sorted(base_ticks, key=lambda z: z[0]):
        if x in seen:
            continue
        seen.add(x)
        tick_vals.append(x)
        tick_text.append(t.replace(" ", "\n") if "\n" not in t else t)

    fig.update_xaxes(
        title=dict(text="Race Distance", font=dict(size=25)),
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=False,
        zeroline=False,
    )

    # separators between legs
    for x in [structure["swim_end"], structure["bike_end"]]:
        fig.add_vline(
            x=x,
            line=dict(color="#AAAAAA", width=1, dash="dot"),
            layer="below",
        )

    # Layout
    fig.update_layout(
        title=dict(text=""),
        annotations=[],
        paper_bgcolor="#7A7A7A",
        plot_bgcolor="#7A7A7A",
        font=dict(color="#FFFFFF"),
        showlegend=False,
        height=650,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    return fig


# --------------------------------------------------
# Render (Streamlit)
# --------------------------------------------------
def render_rank_progression_all_athletes():
    base_title = "Rank Progression – Athletes"

    # year from session state 
    if "selected_year" in st.session_state:
        year = st.session_state["selected_year"]
    elif "year_filter" in st.session_state:
        year = st.session_state["year_filter"]
    else:
        year = "All"

    if year == "All":
        st.subheader(base_title)
        st.info("Please select a single year to view rank progression.")
        return

    year_int = int(year)
    group = _get_group_from_session_state()  

    # Header + info
    header_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with header_col:
        st.subheader(f"{base_title} ({year_int})")
    with info_col:
        with st.popover("ℹ️"):
            structure = race_structure_for_year(year_int)
            st.markdown(
                f"""
**What is shown?**  
Athletes’ **rank over distance** up to the **decision cut-off**.

**This year:** Swim 3.8 km → Bike 180 km → Run {structure["run_km"]} km  
Total: **{structure["run_end"]:.1f} km**

**Group filter:** **{group}**
"""
            )

    df_long = load_long_data()
    df_year = df_long[df_long["year"] == year_int].copy()

    if df_year.empty:
        st.warning(f"No data for year {year_int}.")
        return

    fig = create_rank_progression_figure(
        df_year=df_year,
        year_int=year_int,
        group=group,
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_rank_progression_all_athletes()

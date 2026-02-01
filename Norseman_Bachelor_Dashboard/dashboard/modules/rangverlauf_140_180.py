import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# Path to long-format data file 
# --------------------------------------------------

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

    df["overall_rank_num"] = pd.to_numeric(df["overall_rank"], errors="coerce")
    df = df.dropna(subset=["overall_rank_num", "split_rank", "race_distance_km"])

    df["overall_rank_num"] = df["overall_rank_num"].astype(int)
    df["split_rank"] = df["split_rank"].astype(int)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if "finish_type" in df.columns:
        df["finish_type"] = df["finish_type"].astype(str)

    return df



def get_line_color_for_finisher(finish_type: str) -> str:
    """
    Assign line color based on Black/White finish.
    Any unknown finish type defaults to white.
    """
    t = (finish_type or "").lower()
    if "black" in t:
        return "#000000"   
    if "white" in t:
        return "#FFFFFF"   
    return "#FFFFFF"        


def create_figure_140_180(df: pd.DataFrame, title: str, cutoff_km: float) -> go.Figure:
    """
    Assumes df is already filtered by year and distance cutoff.
    Shows:
      - Finishers whose final rank is 140–180 (bold white/black lines),
      - Athletes who were temporarily in this rank window (thin lines).
    Only segments where split_rank is between 140–180 are drawn.
    """

    rank_min, rank_max = 140, 180

    cols = ["bib", "name", "overall_rank_num"]
    if "finish_type" in df.columns:
        cols.append("finish_type")

    # Finishers ranked 140–180 
    finisher_athletes = (
        df[cols]
        .drop_duplicates(subset=["bib"])
        .query("@rank_min <= overall_rank_num <= @rank_max")
        .sort_values("overall_rank_num")
    )

    # Athletes who were ever in the rank window 
    in_window_mask = df["split_rank"].between(rank_min, rank_max)
    in_window_athletes = df.loc[in_window_mask, cols].drop_duplicates(subset=["bib"])

    # Those who were in the window but did NOT finish 140–180
    intermediate_only_athletes = in_window_athletes[
        ~in_window_athletes["bib"].isin(finisher_athletes["bib"])
    ]

    if finisher_athletes.empty and intermediate_only_athletes.empty:
        fig = go.Figure()
        fig.update_layout(
            title=None,
            paper_bgcolor="#7A7A7A",
            plot_bgcolor="#7A7A7A",
            font=dict(color="#FFFFFF"),
            showlegend=False,
            height=550,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        fig.add_annotation(
            text=title + " – no athletes in this rank window",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#FFFFFF"),
            align="center",
        )
        return fig

    # Get all bibs that should be drawn
    relevant_bibs = pd.concat(
        [finisher_athletes["bib"], intermediate_only_athletes["bib"]]
    ).unique()

    merged = df[df["bib"].isin(relevant_bibs)].copy()

    fig = go.Figure()

    # 1) Athletes who were temporarily in 140–180 
    for _, a in intermediate_only_athletes.iterrows():
        bib = a["bib"]
        name = a["name"]

        df_a = merged[merged["bib"] == bib].sort_values("race_distance_km")

        df_plot = df_a[
            df_a["split_rank"].between(rank_min, rank_max)
        ]
        if df_plot.empty:
            continue

        finish_type = a.get("finish_type", "")
        line_color = get_line_color_for_finisher(finish_type)

        # Thin full-color line 
        fig.add_trace(
            go.Scatter(
                x=df_plot["race_distance_km"],
                y=df_plot["split_rank"],
                mode="lines+markers",
                name=f"{a['overall_rank_num']}. {name} (#{bib})",
                line=dict(width=1.5, color=line_color),
                marker=dict(size=4, color=line_color),
                showlegend=False,
            )
        )

    #  2) Final finishers 140–180 (bold lines) 
    for _, a in finisher_athletes.iterrows():
        bib = a["bib"]
        name = a["name"]
        rank_overall = int(a["overall_rank_num"])

        df_a = merged[merged["bib"] == bib].sort_values("race_distance_km")

        df_plot = df_a[
            df_a["split_rank"].between(rank_min, rank_max)
        ]
        if df_plot.empty:
            continue

        finish_type = a.get("finish_type", "")
        line_color = get_line_color_for_finisher(finish_type)

        fig.add_trace(
            go.Scatter(
                x=df_plot["race_distance_km"],
                y=df_plot["split_rank"],
                mode="lines+markers",
                name=f"{rank_overall}. {name} (#{bib})",
                line=dict(width=3, color=line_color),
                marker=dict(size=8, color=line_color),
                showlegend=False,
            )
        )

    # Y-axis
    fig.update_yaxes(
    title=dict(text="Rank", font=dict(size=25)),
    autorange="reversed",
    tickmode="linear",
    dtick=2,
    range=[rank_max + 0.5, rank_min - 0.5],
    gridcolor="#555555",
    zeroline=False,
)

    # X-axis ticks including the final checkpoint (cutoff_km)
    tick_vals = list(range(0, int(cutoff_km) + 1, 25))
    if cutoff_km not in tick_vals:
        tick_vals.append(cutoff_km)

    fig.update_xaxes(
    title=dict(text="Race Distance (km)", font=dict(size=25)),
    showgrid=True,
    gridcolor="#555555",
    zeroline=False,
    tickmode="array",
    tickvals=tick_vals,
    ticktext=[str(v) for v in tick_vals],
)

    # Layout (no chart title; background in #7A7A7A)
    fig.update_layout(
        title=dict(text=""),
        annotations=[],
        paper_bgcolor="#7A7A7A",
        plot_bgcolor="#7A7A7A",
        font=dict(color="#FFFFFF"),
        showlegend=False,
        height=550,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    return fig


def render_rank_progression_140_180():
    base_title = "Rank Progression – Places 140–180"

    if "selected_year" in st.session_state:
        year = st.session_state["selected_year"]
    elif "year_filter" in st.session_state:
        year = st.session_state["year_filter"]
    else:
        year = "All"

    if year == "All":
        st.subheader(base_title)
        st.info("Please select a single year to view ranks 140–180.")
        return

    year_int = int(year)
    title_with_year = f"{base_title} ({year_int})"

    # --- Header row: title left, info icon right ---
    header_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with header_col:
        st.subheader(title_with_year)
    with info_col:
        with st.popover("ℹ️"):
            st.markdown(
                """
**What is shown in this chart?**

This visualization shows how athletes’ **race positions (ranks)** evolve **within the rank window 140–180**
across the Norseman course.

- **Bold lines**: Athletes who **finished overall rank 140–180**  
- **Thin lines**: Athletes who were **temporarily** in ranks 140–180 but finished outside this window  
- **Line color**: **Black / White** indicates finish category  

**How to interpret this chart**

- Lines moving **up** indicate improving rank  
- Lines moving **down** indicate losing positions  
- Dense crossings suggest high variability and position battles in this part of the field  

**How this can be used**

- Identify course sections where ranks in this window are most volatile  
- Understand when athletes typically move into or out of the 140–180 band  
- Compare how stable this mid-pack rank range is across years

**💡**
You can zoom into the diagram by clicking and dragging a rectangle over the area of interest.     """
            )

    df_long = load_long_data()
    df_long = df_long[df_long["year"] == year_int]

    if df_long.empty:
        st.warning(f"No data for year {year_int}.")
        return

    # ---------------- Distance cutoff ----------------
    # Standard: Stavsro cutoff (37.5 km run) -> 221.3 km total
    if year_int == 2022:
        cutoff_km = 216.3
    else:
        cutoff_km = 221.3

    df_long = df_long[df_long["race_distance_km"] <= cutoff_km]

    fig = create_figure_140_180(
        df_long,
        title=title_with_year,
        cutoff_km=cutoff_km,
    )

    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    render_rank_progression_140_180()

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
# High-contrast color palette for Top 10
# --------------------------------------------------
TOP10_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#17BECF", "#BCBD22", "#E377C2", "#8C564B", "#AEC7E8",
]

# Dark grey for "contenders" (temporarily in Top 10)
CONTENDER_COLOR = "rgba(60, 60, 60, 0.85)"

# Gridlines (parallel to X axis) a bit brighter
Y_GRID_COLOR = "rgba(255,255,255,0.25)"

# Legend font size
LEGEND_FONT_SIZE = 14

# --------------------------------------------------
# For Norwegian Mojibake
# --------------------------------------------------
def fix_norwegian_mojibake(s):
    if pd.isna(s):
        return s
    s = str(s)

    replacements = {
        "√©": "é", "√®": "®", "√™": "™", "√•": "å", "√¶": "ö",
        "√∏": "ø", "√∂": "ø",
        "√º": "å", "√Ñ": "Å", "√Å": "Å",
        "√ª": "æ",
        "√Ä": "Ä", "√Ö": "Ö", "√Ü": "Ü",
        "√∫": "ú", "√ú": "ú", "√ù": "ù",
        "√í": "í", "√ó": "ó", "√ò": "ò",
        "√ß": "ß", "√±": "ñ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_long_data() -> pd.DataFrame:
    if data_store is None:
        raise RuntimeError("data_store not available. Place data_store.py next to main.py.")

    df = data_store.df_long()

    if "name" in df.columns:
        df["name"] = df["name"].apply(fix_norwegian_mojibake)

    df["overall_rank_num"] = pd.to_numeric(df["overall_rank"], errors="coerce")
    df = df.dropna(subset=["split_rank", "race_distance_km"])
    df["split_rank"] = df["split_rank"].astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    sort_cols = ["year", "bib", "race_distance_km"]
    if "cum_time_seconds" in df.columns:
        sort_cols.append("cum_time_seconds")
    elif "split_time_td" in df.columns:
        sort_cols.append("split_time_td")

    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(subset=["year", "bib", "race_distance_km"], keep="first")

    return df



# --------------------------------------------------
# Helper for contiguous segments
# --------------------------------------------------
def _contiguous_segments(df_a: pd.DataFrame, flag_col: str):
    segments, current = [], []
    for _, row in df_a.iterrows():
        if row[flag_col]:
            current.append(row)
        else:
            if current:
                segments.append(pd.DataFrame(current))
                current = []
    if current:
        segments.append(pd.DataFrame(current))
    return segments


# --------------------------------------------------
# Figure generator
# --------------------------------------------------
def create_top10_figure(
    df: pd.DataFrame,
    focus_top10: bool,
    title: str,
    finish_km: float,
) -> go.Figure:
    
    df = df.copy()

    dist_in_plot = df["race_distance_km"].where(df["race_distance_km"].le(finish_km))
    max_dist_in_plot = dist_in_plot.groupby(df["bib"]).transform("max")
    is_last_plotted = df["race_distance_km"].eq(max_dist_in_plot)

    mask = is_last_plotted & df["overall_rank_num"].between(1, 10)
    df.loc[mask, "split_rank"] = df.loc[mask, "overall_rank_num"].astype(int)

    df["in_top10"] = df["split_rank"].between(1, 10)

    finishers = (
        df[["bib", "name", "overall_rank_num"]]
        .drop_duplicates("bib")
        .query("1 <= overall_rank_num <= 10")
        .sort_values("overall_rank_num")
    )

    if finishers.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=""),
            paper_bgcolor="#7A7A7A",
            plot_bgcolor="#7A7A7A",
            font=dict(color="#FFFFFF"),
        )
        fig.add_annotation(
            text=title + " – no Top-10 finishers in this year",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#FFFFFF"),
        )
        return fig

    fig = go.Figure()

    #  Focus mode (only Top-10 phases) 
    if focus_top10:

        #  Top-10 finishers segments 
        for _, a in finishers.iterrows():
            bib = a["bib"]
            name = a["name"]
            rank_idx = int(a["overall_rank_num"]) - 1
            color = TOP10_COLORS[rank_idx % len(TOP10_COLORS)]

            df_a = df[df["bib"] == bib].sort_values("race_distance_km").copy()
            df_a["in_range"] = df_a["in_top10"]

            for i, seg in enumerate(_contiguous_segments(df_a, "in_range")):
                seg = seg[(seg["race_distance_km"] >= 0) & (seg["race_distance_km"] <= finish_km)]
                if seg.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=seg["race_distance_km"],
                        y=seg["split_rank"],
                        mode="lines+markers",
                        name=f"{a['overall_rank_num']}. {name} (#{bib})" if i == 0 else "",
                        showlegend=i == 0,
                        line=dict(width=3, color=color),
                        marker=dict(size=8, color=color),
                    )
                )

        #  Contenders (temporarily Top 10, but not Top-10 finishers) 
        contender_bibs = df.loc[df["in_top10"], "bib"].dropna().unique()
        contender_bibs = [b for b in contender_bibs if b not in finishers["bib"].values]

        for bib in contender_bibs:
            df_a = df[df["bib"] == bib].sort_values("race_distance_km").copy()
            df_a["in_range"] = df_a["in_top10"]

            for seg in _contiguous_segments(df_a, "in_range"):
                seg = seg[(seg["race_distance_km"] >= 0) & (seg["race_distance_km"] <= finish_km)]
                if seg.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=seg["race_distance_km"],
                        y=seg["split_rank"],
                        mode="lines+markers",
                        line=dict(width=2, color=CONTENDER_COLOR),
                        marker=dict(size=6, color=CONTENDER_COLOR),
                        showlegend=False,
                        hovertemplate=(
                            f"#{bib}<br>"
                            "km=%{x}<br>"
                            "rank=%{y}<extra></extra>"
                        ),
                    )
                )

        fig.update_yaxes(
            title=dict(text="Rank", font=dict(size=25)),
            autorange=False,
            range=[10.5, 0.5],
            dtick=1,
            showgrid=True,
            gridcolor=Y_GRID_COLOR,
        )

    #  Expanded mode (full race progression for Top-10 finishers) 
    else:
        for _, a in finishers.iterrows():
            bib = a["bib"]
            name = a["name"]
            rank_idx = int(a["overall_rank_num"]) - 1
            color = TOP10_COLORS[rank_idx % len(TOP10_COLORS)]

            df_a = df[df["bib"] == bib].sort_values("race_distance_km").copy()
            df_a = df_a[(df_a["race_distance_km"] >= 0) & (df_a["race_distance_km"] <= finish_km)]
            if df_a.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=df_a["race_distance_km"],
                    y=df_a["split_rank"],
                    mode="lines+markers",
                    name=f"{a['overall_rank_num']}. {name} (#{bib})",
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color),
                )
            )

        df_top10 = df[df["bib"].isin(finishers["bib"])]
        max_rank_top10 = max(int(df_top10["split_rank"].max()), 10)

        fig.update_yaxes(
            title=dict(text="Rank", font=dict(size=25)),
            autorange=False,
            tickmode="linear",
            tick0=1,
            dtick=5,
            range=[max_rank_top10 + 0.5, 0.5],
            showgrid=True,
            gridcolor=Y_GRID_COLOR,
        )

    #  X axis: hard clip so nothing renders before 0 
    swim_end = 3.8
    bike_end = 3.8 + 180.0
    run_km = max(0.0, round(finish_km - bike_end, 1))

    tick_vals = [
        0.0,
        swim_end,
        swim_end + 47.0,
        swim_end + 90.0,
        swim_end + 142.0,
        bike_end,
        bike_end + 20.0,
        finish_km,
    ]
    tick_text = [
        " ",
        "Swim\n3.8 km",
        "Bike\n47",
        "Bike\n90",
        "Bike\n142",
        "Bike\n180 km",
        "Run\n20",
        "Run\n42.2 km",
    ]

    fig.update_xaxes(
        title=dict(text="Race Distance (km)", font=dict(size=25)),
        range=[0, finish_km],
        constrain="range",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=False,
        zeroline=False,
        tickangle=0,
    )

    for x in [swim_end, bike_end]:
        fig.add_vline(
            x=x,
            line=dict(color="#AAAAAA", width=1, dash="dot"),
            layer="below",
        )

    fig.update_layout(
        title=dict(text=""),
        legend_title="Athlete",
        legend=dict(
            font=dict(size=LEGEND_FONT_SIZE),
            title=dict(font=dict(size=LEGEND_FONT_SIZE)),
        ),
        height=550,
        paper_bgcolor="#7A7A7A",
        plot_bgcolor="#7A7A7A",
        margin=dict(l=40, r=20, t=20, b=40),
        font=dict(color="#FFFFFF"),
    )

    return fig


# --------------------------------------------------
# Streamlit Entry Point
# --------------------------------------------------
def render_top10_rank_progression():
    base_title = "Rank Progression – Top 10"

    year = st.session_state.get("selected_year") or st.session_state.get("year_filter")
    if year is None or year == "All":
        st.info("Please select a single year.")
        return

    year = int(year)
    title_with_year = f"{base_title} ({year})"

    # finish km: 2022 is shorter
    finish_km = 216.3 if year == 2022 else 221.3

    header_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with header_col:
        st.subheader(title_with_year)
    with info_col:
        with st.popover("ℹ️"):
            st.markdown("""
**What is shown in this chart?**

This visualization shows how the **Top 10** athletes **race positions (ranks)** evolve throughout the Norseman course

- **X-axis**: accumulated distance along the course (km)  
- **Y-axis**: rank at each official timing station (1 = best)  

- **Colored lines**: athletes who **finished overall rank 1–10**  
- **Dark grey lines**: athletes who were **temporarily in the Top 10** (only when *Show only Top-10 places* is enabled)

**How to interpret this chart**

- Lines moving **up** indicate improving position.  
- Lines moving **down** indicate losing positions.  
- Line crossings highlight **overtakes** and position battles.

**How this can be used**

- Identify **where decisive moves happen** among the Top 10.  
- See whether the Top 10 **stabilizes early** or stays volatile.  
- Compare tactical patterns and turning points **across different years**.
            """)

    df_year = load_long_data().query("year == @year")

    # Checkbox 
    focus_top10 = st.checkbox("Focus on 10 places", value=True)

    fig = create_top10_figure(df_year, focus_top10, title_with_year, finish_km=finish_km)
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    render_top10_rank_progression()

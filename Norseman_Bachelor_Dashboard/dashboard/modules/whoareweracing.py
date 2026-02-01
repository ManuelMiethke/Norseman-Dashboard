import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_who_are_we_racing(df: pd.DataFrame) -> None:
    # ==================================================
    # Header + Info 
    # ==================================================
    col_title, col_info = st.columns([0.92, 0.08], vertical_alignment="center")

    with col_title:
        st.markdown("## Who Are We Racing?")

    with col_info:
        with st.popover("ℹ️"):
            st.markdown("""
**What is shown in this chart?**

This stacked bar chart shows how many athletes ended up in each **outcome group** per year.

- **X-axis**: race year  
- **Y-axis**: number of athletes (starters)  

**Outcome groups**

- **Black Shirts**: finished Black and ended on Mount Gaustatoppen   
- **Missed (due to Competition)**: reached the year-specific run cut-off **within 14:45h** but had **rank > 160** at that cut-off  
- **White Finishers (no cut-off)**: finished White but were **not** in the in-time-but-rank>160 group  
- **DNF**: hardcoded DNF counts per year  

**How to interpret**

- Larger **Black** → more Black-shirt eligible athletes  
- Larger **Missed (Competition)** → stronger pressure at the cut-off frontier  
- Larger **White (no cut-off)** → more White finishers outside the “missed by rank” group  
- Larger **DNF** → more athletes did not finish that year
            """)

    # --------------------------------------------------
    # 0. Sicherheitschecks
    # --------------------------------------------------
    required_cols = {
        "year",
        "bib",
        "split_key",
        "cum_time_seconds",
        "split_rank",
    }
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required columns for WhoAreWeRacing: {missing}")
        return

    # --------------------------------------------------
    # 0.1 Jahre ohne Rennen raus + keine Achsen-Lücken
    # --------------------------------------------------
    YEARS_TO_PLOT = [2018, 2019, 2021, 2022, 2024, 2025]
    df = df[df["year"].isin(YEARS_TO_PLOT)].copy()

    # --------------------------------------------------
    # 1. Jahr-spezifischen Cut-off Split festlegen
    # --------------------------------------------------
    CUTOFF_SPLIT_BY_YEAR = {
        2018: "run_32_5km_langefonn",
        2019: "run_32_5km_langefonn",
        2021: "run_32_5km_langefonn",
        2022: "run_32_5km_langefonn",
        2024: "run_37_5km_stavsro_cut_off",
        2025: "run_37_5km_stavsro_cut_off",
    }

    df_cutoff_parts = []
    for y, key in CUTOFF_SPLIT_BY_YEAR.items():
        df_y = df[(df["year"] == y) & (df["split_key"].astype(str) == key)]
        df_cutoff_parts.append(df_y)

    df_cutoff = (
        pd.concat(df_cutoff_parts, ignore_index=True)
        if df_cutoff_parts
        else df.iloc[0:0].copy()
    )

    # Zeit-Cutoff (14h45)
    CUTOFF_SECONDS = 14 * 3600 + 45 * 60

    # --------------------------------------------------
    # 1.1 Finish-Splits: finish_type[black/white] robust aus split_key
    # --------------------------------------------------
    sk = df["split_key"].astype(str).str.lower()
    is_white_finish_split = sk.str.contains("finish", na=False) & sk.str.contains("white", na=False)
    is_black_finish_split = sk.str.contains("finish", na=False) & sk.str.contains("black", na=False)

    df_white_finish = df[is_white_finish_split].copy()
    df_black_finish = df[is_black_finish_split].copy()

    # --------------------------------------------------
    # 1.2 Hardcoded DNF counts
    # --------------------------------------------------
    DNF_HARDCODED = {
        2018: 7,
        2019: 12,
        2021: 11,
        2022: 16,
        2024: 17,
        2025: 11,
    }

    results = []

    # --------------------------------------------------
    # 2. Aggregation pro Jahr
    # --------------------------------------------------
    for year, g_year in df.groupby("year"):
        g_cut = df_cutoff[df_cutoff["year"] == year]
        g_wfinish = df_white_finish[df_white_finish["year"] == year]
        g_bfinish = df_black_finish[df_black_finish["year"] == year]

        finish_white = g_wfinish["bib"].nunique()
        finish_black = g_bfinish["bib"].nunique()

        if g_cut.empty:
            white_comp = 0
        else:
            within_time = g_cut["cum_time_seconds"] <= CUTOFF_SECONDS
            ranks = g_cut["split_rank"]
            comp_mask = within_time & (ranks > 160)
            white_comp = g_cut.loc[comp_mask, "bib"].nunique()

        white_other = finish_white - white_comp
        if white_other < 0:
            white_other = 0

        dnf = int(DNF_HARDCODED.get(int(year), 0))

        results.append(
            {
                "year": int(year),
                "black": finish_black,
                "white_comp": white_comp,
                "white_other": white_other,
                "dnf": dnf,
            }
        )

    df_plot = pd.DataFrame(results).sort_values("year")
    df_plot["year_str"] = df_plot["year"].astype(str)
    year_order = df_plot["year_str"].tolist()

    # --------------------------------------------------
    # 3. Gestapeltes Balkendiagramm
    # --------------------------------------------------
    DARK_GRAY = "black"
    BG = "#7A7A7A"
    GRID = "rgba(0,0,0,0.12)"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_plot["year_str"],
            y=df_plot["dnf"],
            name="DNF",
            marker_color="red",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["year_str"],
            y=df_plot["white_other"],
            name="White Finishers (no cut-off)",
            marker=dict(color="white", line=dict(color="black", width=1)),
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["year_str"],
            y=df_plot["white_comp"],
            name="Missed (due to competition)",
            marker=dict(
                color="white",
                line=dict(color="black", width=1),
                pattern=dict(shape="/", fgcolor="black", bgcolor="white", solidity=0.35),
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["year_str"],
            y=df_plot["black"],
            name="Black Shirts",
            marker_color=DARK_GRAY,
        )
    )

    fig.update_layout(
        barmode="stack",
        bargap=0.35,
        title=dict(
            text="Black Shirt Outcomes by Year (Cut-off Rank vs Time/DNF)",
            x=0.1,
            font=dict(size=26),
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=22)),
            type="category",
            categoryorder="array",
            categoryarray=year_order,
            tickfont=dict(size=18),
            showgrid=True,
            gridcolor=GRID,
        ),
        yaxis=dict(
            title=dict(text="Number of Athletes", font=dict(size=22)),
            tickfont=dict(size=18),
            showgrid=True,
            gridcolor=GRID,
        ),
        legend=dict(
            title=dict(text="Outcome Type", font=dict(size=18)),
            font=dict(size=16),
        ),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color="black", size=16),
    )

    st.plotly_chart(fig, width="stretch")

    # ==================================================
    # Header + Info 
    # ==================================================
    col_title2, col_info2 = st.columns([0.92, 0.08], vertical_alignment="center")

    with col_title2:
        st.markdown("### Empirical Black Shirt Pace (Cut-off Frontier)")

    with col_info2:
        with st.popover("ℹ️"):
            st.markdown("""
**What is shown in this chart?**

This horizontal bar chart shows the **empirical Black Shirt cut-off frontier**
at the **37.5 km run cut-off (Stavsro)** for **2024 and 2025**.

- **Y-axis**: year  
- **X-axis**: elapsed time (hours) at the cut-off for **rank ~160**

**How to interpret**

- **Shorter bar** → stronger competition (faster cut-off)  
- **Longer bar** → weaker competition (slower cut-off)  
- **Red dotted line** → median of 2024–2025

**How this can be used**

- Benchmark Black Shirt cut-off pressure  
- Compare competitiveness between years  
- Provide realistic performance targets
            """)

    # --------------------------------------------------
    # 4. Empirical Black Shirt Frontier 
    # --------------------------------------------------
    if df_cutoff.empty:
        st.info("No cut-off data available to compute frontier.")
        return

    frontier_years = [2024, 2025]
    df_cutoff_375 = df_cutoff[df_cutoff["year"].isin(frontier_years)].copy()

    if df_cutoff_375.empty:
        st.info("No 37.5 km cut-off data available for 2024/2025.")
        return

    def get_frontier_time(group: pd.DataFrame):
        g = group.dropna(subset=["split_rank", "cum_time_seconds"]).copy()
        if g.empty:
            return None

        g160 = g[g["split_rank"] == 160]
        if not g160.empty:
            return g160["cum_time_seconds"].iloc[0]

        idx = (g["split_rank"] - 160).abs().idxmin()
        return g.loc[idx, "cum_time_seconds"]

    frontier_series = (
        df_cutoff_375.groupby("year", group_keys=False)
        .apply(get_frontier_time, include_groups=False)
        .dropna()
    )

    if frontier_series.empty:
        st.info("No cut-off rank data available to compute frontier for 2024/2025.")
        return

    frontier_hours = (frontier_series / 3600.0).sort_index()
    median_frontier = frontier_hours.median()

    st.markdown(
        f"**Median time of rank ~160 at the 37.5 km cut-off (2024–2025): {median_frontier:.2f} hours**"
    )

    frontier_df = pd.DataFrame(
        {"year": frontier_hours.index.astype(int), "hours": frontier_hours.values}
    )

    frontier_df["year_str"] = frontier_df["year"].astype(str)
    frontier_df = frontier_df.sort_values("year", ascending=False)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            y=frontier_df["year_str"],
            x=frontier_df["hours"],
            orientation="h",
            marker=dict(color=DARK_GRAY),
            width=0.35,
            name="Rank ~160 time",
        )
    )

    fig2.add_vline(
        x=float(median_frontier),
        line_width=4,
        line_dash="dot",
        line_color="red",
    )

    fig2.update_layout(
        title=dict(
            text="Black Shirt Frontier: Time of rank ~160 at 37.5 km cut-off (2024–2025)",
            x=0.1,
            font=dict(size=26),
        ),
        xaxis=dict(
            title=dict(text="Hours at cut-off point", font=dict(size=22)),
            tickfont=dict(size=18),
            showgrid=True,
            gridcolor=GRID,
        ),
        yaxis=dict(
            title=dict(text="Year", font=dict(size=22)),
            tickfont=dict(size=18),
            type="category",
            categoryorder="array",
            categoryarray=["2024", "2025"],
            showgrid=False,
        ),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color="black", size=16),
        showlegend=False,
    )

    st.plotly_chart(fig2, width="stretch")

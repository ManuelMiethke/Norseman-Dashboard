import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_store import df_long


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _split_order_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply split rules:
    - remove finish splits (black/white finish) from heatmap
    - year-dependent run cut (based on cumulative race_distance_km)
    - build split order by increasing cumulative distance (race order)
    """
    d = df.copy()

    # Remove black/white finish splits from diagram (keep swim_finish/bike_180km_finish!)
    finish_black = "finish_black_t_shirt"
    finish_white = "finish_white_t_shirt"
    d = d[~d["split_key"].isin([finish_black, finish_white])].copy()

    years_32 = {2018, 2019, 2021, 2022}
    years_37 = {2024, 2025}

    if "race_distance_km" in d.columns:
        d["race_distance_km"] = pd.to_numeric(d["race_distance_km"], errors="coerce")

        cutoff_32_cum = 183.8 + 32.5  # 216.3
        cutoff_37_cum = 183.8 + 37.5  # 221.3

        def _keep_row(row) -> bool:
            sk = str(row["split_key"])
            if not sk.startswith("run_"):
                return True
            y = int(row["year"])
            dist = row["race_distance_km"]
            if pd.isna(dist):
                return True
            if y in years_32:
                return dist <= cutoff_32_cum
            if y in years_37:
                return dist <= cutoff_37_cum
            return True

        d = d[d.apply(_keep_row, axis=1)].copy()

        split_order = (
            d.groupby("split_key", as_index=False)
            .agg(dist=("race_distance_km", "median"))
            .sort_values(["dist"], na_position="last")["split_key"]
            .tolist()
        )
    else:
        split_order = sorted(d["split_key"].dropna().unique().tolist())

    d["split_key"] = pd.Categorical(d["split_key"], categories=split_order, ordered=True)
    return d


def _robust_absmax(z: np.ndarray, q: float = 0.98) -> float:
    """Robust scale to prevent outliers from washing out the heatmap."""
    zflat = z[np.isfinite(z)]
    if zflat.size == 0:
        return 1.0
    return float(np.quantile(np.abs(zflat), q))


def _cat_from_athlete_key(k: str) -> str:
    # athlete_key format: "<category> | <bib> | <year>"
    return str(k).split(" | ")[0].strip()


# ------------------------------------------------------------
# Public render function
# ------------------------------------------------------------
def render_pacing_heatmap(selected_year, selected_group):
    # --- Header + info popover
    title_with_year = "Pacing Heatmap (Δ Zeit % vs. Median je Split)"
    header_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")
    with header_col:
        st.subheader(title_with_year)
    with info_col:
        with st.popover("ℹ️"):
            st.markdown("""
**What is shown in this chart?**

This heatmap shows, for each timing station (**split**), how much faster or slower each athlete was **relative to the median time at that split**.

- **Rows (Y-axis)**: official timing stations (splits) in race order  
- **Columns (X-axis)**: athletes, grouped by finish rank from left to right: **Top 10**, **Black Shirt**, **White Shirt**, **DNF**  
- **Color**: **Δ time (%) vs. split median**  
  - **Orange** = faster than median (negative Δ%)  
  - **Blue** = slower than median (positive Δ%)

**How to interpret this chart**

- A consistent orange/blue pattern across many rows indicates a stable pacing profile.  
- Strong color changes across rows suggest pacing shifts, segment-specific strengths/weaknesses, or fatigue effects.

**How this can be used**

- Compare pacing consistency across finisher groups.  
- Identify where gaps between groups emerge (bike climbs, late run segments, etc.).  
- Spot atypical pacing patterns that may relate to DNF outcomes.
            """)

    df = df_long()

    required = {"split_key", "segment_time_s", "overall_rank", "finish_type", "name", "bib", "year"}
    missing = sorted(required - set(df.columns))
    if missing:
        st.error(f"Es fehlen benötigte Spalten in df_long(): {missing}")
        st.stop()

    d = df.copy()

    # numeric
    d["segment_time_s"] = pd.to_numeric(d["segment_time_s"], errors="coerce")

    # swim_finish should represent start -> swim_finish (cum time)
    if "cum_time_seconds" in d.columns:
        mask_swim = d["split_key"] == "swim_finish"
        d.loc[mask_swim, "segment_time_s"] = pd.to_numeric(d.loc[mask_swim, "cum_time_seconds"], errors="coerce")

    # Header filters
    if selected_year != "All":
        d = d[d["year"] == selected_year].copy()

    # Categories
    d["category"] = np.where(
        pd.to_numeric(d["overall_rank"], errors="coerce") <= 10,
        "Top 10",
        d["finish_type"].astype(str),
    )
    finish_map = {"Black": "Black Shirt", "White": "White Shirt", "DNF": "DNF", "Top 10": "Top 10"}
    d["category"] = d["category"].replace(finish_map)

    if selected_group != "All":
        d = d[d["category"] == selected_group].copy()

    if d.empty:
        st.warning("Keine Daten nach den gewählten Filtern.")
        st.stop()

    # Split rules + ordering + year-dependent run cut
    d = _split_order_and_filter(d)

    # Remove transition-like markers + remove the imputed/white split explicitly
    transition_splits = {"bike_start", "run_start", "transition_1_in"}
    imputed_splits = {"bike_152km_end_imingfjell"}  # <- remove this one
    drop_splits = transition_splits.union(imputed_splits)

    d = d[~d["split_key"].astype(str).isin(drop_splits)].copy()

    if d.empty:
        st.warning("Keine Daten nach dem Entfernen der Splits.")
        st.stop()

    # X-axis category order
    cat_levels = ["Top 10", "Black Shirt", "White Shirt", "DNF"]
    d["cat_order"] = pd.Categorical(d["category"], categories=cat_levels, ordered=True)

    # Athlete key + label
    d["athlete_key"] = d["category"].astype(str) + " | " + d["bib"].astype(str) + " | " + d["year"].astype(str)
    d["athlete_label"] = d["category"].astype(str)

    d["segment_time_s"] = pd.to_numeric(d["segment_time_s"], errors="coerce").astype(float)

    # Δ Zeit (%) vs Median 
    median_t = d.groupby("split_key", observed=False)["segment_time_s"].transform("median")
    denom = median_t.replace(0, np.nan)
    d["delta_pct"] = (d["segment_time_s"] - median_t) / denom * 100.0

    # Athlete ordering
    d["overall_rank_num"] = pd.to_numeric(d["overall_rank"], errors="coerce")
    athlete_order = (
        d[["athlete_key", "athlete_label", "cat_order", "overall_rank_num"]]
        .drop_duplicates()
        .sort_values(["cat_order", "overall_rank_num", "athlete_key"])["athlete_key"]
        .tolist()
    )

    # Counts per category
    athletes_meta = d[["athlete_key", "athlete_label", "cat_order"]].drop_duplicates().copy()
    counts = athletes_meta["athlete_label"].value_counts().to_dict()
    xaxis_title = (
        "Athleten ("
        + "Top 10: " + str(int(counts.get("Top 10", 0)))
        + " | Black Shirt: " + str(int(counts.get("Black Shirt", 0)))
        + " | White Shirt: " + str(int(counts.get("White Shirt", 0)))
        + " | DNF: " + str(int(counts.get("DNF", 0)))
        + ")"
    )

    # Heat matrix
    split_order = list(d["split_key"].cat.categories)
    split_order = [sk for sk in split_order if str(sk) not in drop_splits]

    heat = (
        d.pivot_table(
            index="split_key",
            columns="athlete_key",
            values="delta_pct",
            aggfunc="mean",
        )
        .reindex(index=split_order)
        .reindex(columns=athlete_order)
    )

    # ------------------------------------------------------------
    # Robust color scaling (percent) - exclude DNF + aggressive quantile + cap
    # ------------------------------------------------------------
    cols_no_dnf = [c for c in heat.columns if _cat_from_athlete_key(c) != "DNF"]
    base_vals = heat[cols_no_dnf].values if len(cols_no_dnf) > 0 else heat.values

    absmax = _robust_absmax(base_vals, q=0.90)  # boost contrast
    absmax = max(absmax, 8.0)                   # avoid over-amplifying tiny noise
    absmax = min(absmax, 30.0)                  # cap extremes so Top10 becomes saturated

    colorscale = [
        [0.00, "#8c2d04"],
        [0.10, "#cc4c02"],
        [0.20, "#ec7014"],
        [0.30, "#fe9929"],
        [0.40, "#fec44f"],
        [0.50, "#f7f7f7"],
        [0.60, "#c6dbef"],
        [0.70, "#9ecae1"],
        [0.80, "#6baed6"],
        [0.90, "#3182bd"],
        [1.00, "#08519c"],
    ]

    athlete_df = (
        d[["athlete_key", "athlete_label", "cat_order"]]
        .drop_duplicates()
        .set_index("athlete_key")
        .loc[athlete_order]
        .reset_index()
    )

    tickvals, ticktext = [], []
    for cat in athlete_df["athlete_label"].unique():
        idx = athlete_df.index[athlete_df["athlete_label"] == cat]
        if len(idx) == 0:
            continue
        mid = idx[len(idx) // 2]
        tickvals.append(athlete_df.loc[mid, "athlete_key"])
        ticktext.append(cat)

    # Group boundary separators 
    cat_sequence = ["Top 10", "Black Shirt", "White Shirt", "DNF"]
    group_indices = {
        cat: athlete_df.index[athlete_df["athlete_label"] == cat].to_list()
        for cat in cat_sequence
    }
    boundary_keys = []
    for cat in cat_sequence[1:]:
        idxs = group_indices.get(cat, [])
        if len(idxs) > 0:
            boundary_keys.append(athlete_df.loc[idxs[0], "athlete_key"])

    shapes = []
    for xkey in boundary_keys:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=xkey,
                x1=xkey,
                y0=0,
                y1=1,
                line=dict(color="black", width=2),
                layer="above",
            )
        )

    fig = go.Figure(
        go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.astype(str).tolist(),
            zmin=-absmax,
            zmax=absmax,
            colorscale=colorscale,
            colorbar=dict(title="Δ Zeit (%) vs. Median"),
            hoverongaps=False,
            zsmooth=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Split: %{y}<br>"
                "Δ vs. Median: %{z:.1f} %<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=850,
        xaxis_title=xaxis_title,
        yaxis_title="Split",
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            automargin=True,
        ),
        yaxis=dict(autorange=True),
        margin=dict(l=60, r=20, t=10, b=140),
        shapes=shapes,
    )

    st.plotly_chart(fig, use_container_width=True)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_wide_df() -> pd.DataFrame:
    import data_store
    return data_store.df_wide().copy()

SEGMENT_COLORS = {
    "Swim": "#64b5f6",   
    "Bike": "#4A4A4A",   
    "Run (to 32.5 km)": "#2e7d32",    
}
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _hms_to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        parts = s.split(":")
        if len(parts) != 3:
            return np.nan
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + int(sec)
    except Exception:
        return np.nan


def _seconds_to_hms(sec: float) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return "—"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _seconds_to_min_str(sec: float) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return "—"
    return f"{sec/60:.1f} min"


def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


# ------------------------------------------------------------
# Filtering by header group
# ------------------------------------------------------------
def _apply_group_filter(df: pd.DataFrame, selected_group: str) -> pd.DataFrame:
    g = (selected_group or "All").strip()

    # "All" -> no filter
    if g == "All":
        return df

    # Prefer finish_type if present
    finish_col = None
    for c in ["finish_type", "finish_raw", "finish"]:
        if c in df.columns:
            finish_col = c
            break

    if g == "DNF":
        if finish_col:
            return df[df[finish_col].apply(_norm_str).str.contains("dnf")]
        # fallback: nothing we can do safely
        return df.iloc[0:0].copy()

    if g in ["Black Shirt", "White Shirt"]:
        if finish_col:
            key = "black" if g == "Black Shirt" else "white"
            return df[df[finish_col].apply(_norm_str).str.contains(key)]
        return df.iloc[0:0].copy()

    if g == "Top 10":
        # Best effort: overall_rank column
        if "overall_rank" in df.columns:
            return df[pd.to_numeric(df["overall_rank"], errors="coerce") <= 10]
        # fallback: nothing we can do safely
        return df.iloc[0:0].copy()

    return df


# ------------------------------------------------------------
# Compute medians
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _compute_year_medians(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "year" not in d.columns:
        raise ValueError("Expected column 'year' in wide dataset.")

    required = ["swim_time", "bike_time", "run_start_time", "run_32_5km_langefonn_time"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d["swim_sec"] = d["swim_time"].apply(_hms_to_seconds)
    d["bike_sec"] = d["bike_time"].apply(_hms_to_seconds)
    d["run_start_sec"] = d["run_start_time"].apply(_hms_to_seconds)
    d["run_32_5_sec"] = d["run_32_5km_langefonn_time"].apply(_hms_to_seconds)

    # Run segment = time at 32.5 - time at run start
    d["run32_seg_sec"] = d["run_32_5_sec"] - d["run_start_sec"]
    d.loc[d["run32_seg_sec"] <= 0, "run32_seg_sec"] = np.nan

    out = []
    for year, g in d.groupby("year", dropna=True):
        try:
            year = int(year)
        except Exception:
            pass

        out.append({"year": year, "leg": "Swim", "median_sec": np.nanmedian(g["swim_sec"].values)})
        out.append({"year": year, "leg": "Bike", "median_sec": np.nanmedian(g["bike_sec"].values)})
        out.append({"year": year, "leg": "Run (to 32.5 km)", "median_sec": np.nanmedian(g["run32_seg_sec"].values)})

    med = pd.DataFrame(out).dropna(subset=["median_sec"]).copy()
    med["year"] = med["year"].astype(int)
    return med


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def _build_year_compare_chart(med: pd.DataFrame, show_in: str) -> go.Figure:
    years = sorted(med["year"].unique(), reverse=True)
    leg_order = ["Swim", "Bike", "Run (to 32.5 km)"]

    med = med.copy()
    med["leg"] = pd.Categorical(med["leg"], categories=leg_order, ordered=True)
    med = med.sort_values(["year", "leg"])

    # Baseline medians (2025) per leg for compare mode
    base = (
        med[med["year"] == 2025][["leg", "median_sec"]]
        .dropna()
        .set_index("leg")["median_sec"]
        .to_dict()
    )

    fig = go.Figure()

    for leg in leg_order:
        sub = med[med["leg"] == leg].copy()
        secs = sub["median_sec"].tolist()
        y = sub["year"].tolist()

        # X-axis always in hours
        x = [
            (v / 3600.0) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan
            for v in secs
        ]

        if show_in == "Time Overview":
            # Normal mode: show HH:MM:SS, use segment colors
            txt = [_seconds_to_hms(v) for v in secs]
            colors = [SEGMENT_COLORS.get(leg, "#999999")] * len(secs)
        else:
            # Compare mode vs 2025: faster = green, slower = red (2025 neutral)
            txt = []
            colors = []
            for year, s in zip(y, secs):
                if s is None or (isinstance(s, float) and np.isnan(s)) or leg not in base or base[leg] is None or np.isnan(base[leg]):
                    txt.append("—")
                    colors.append("#999999")
                    continue

                delta_sec = float(s) - float(base[leg])
                delta_min = delta_sec / 60.0
                sign = "+" if delta_min > 0 else ""
                txt.append(f"{_seconds_to_hms(s)} ({sign}{delta_min:.1f} min)")

                if year == 2025:
                    colors.append("#bdbdbd")
                else:
                    if delta_sec < 0:
                        colors.append("#2e7d32")  # faster
                    elif delta_sec > 0:
                        colors.append("#c62828")  # slower
                    else:
                        colors.append("#bdbdbd")

        fig.add_trace(
        go.Bar(
        name=leg,
        x=x,
        y=y,
        orientation="h",
        marker_color=colors if show_in != "Time Overview" else SEGMENT_COLORS.get(leg, "#999999"),
        text=txt,
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=(
            "<b>Year</b>: %{y}<br>"
            "<b>Leg</b>: " + leg + "<br>"
            "<b>Median</b>: %{text}<extra></extra>"
        ),
    )
)

    fig.update_layout(
        paper_bgcolor="#7a7a7a",   
        plot_bgcolor="#7a7a7a",
        barmode="group",
        bargap=0.28,
        bargroupgap=0.10,
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            title="Median time (hours)",
            showgrid=True,
            zeroline=False,
            tickformat=".1f",
        ),
        yaxis=dict(
            title="Year",
            type="category",
            categoryorder="array",
            categoryarray=years,
            autorange="reversed",
),
        legend=dict(
            title=" ",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    return fig


def _build_finish_trend_chart(df: pd.DataFrame) -> go.Figure:
    d = df.copy()

    if "year" not in d.columns:
        raise ValueError("Expected column 'year' in wide dataset.")

    # Choose a finish-time column if available
    finish_col = None
    for c in ["finish_time", "finish_total_time", "finish", "total_time", "overall_time"]:
        if c in d.columns:
            finish_col = c
            break

    if finish_col is None:
        # Fall back to sum of legs if explicit finish time is not available
        needed = ["swim_time", "bike_time", "run_time"]
        if not all(c in d.columns for c in needed):
            raise ValueError("No finish time column found and cannot fall back to swim/bike/run_time.")
        d["finish_sec"] = (
            d["swim_time"].apply(_hms_to_seconds)
            + d["bike_time"].apply(_hms_to_seconds)
            + d["run_time"].apply(_hms_to_seconds)
        )
    else:
        d["finish_sec"] = d[finish_col].apply(_hms_to_seconds)

    # Median finish per year
    agg = (
        d.groupby("year", dropna=True)["finish_sec"]
        .median()
        .reset_index()
        .dropna(subset=["finish_sec"])
        .copy()
    )
    agg["year"] = pd.to_numeric(agg["year"], errors="coerce").astype("Int64").astype(int)
    agg = agg.sort_values("year")

    # Convert to hours for axis
    agg["finish_h"] = agg["finish_sec"] / 3600.0
    x_years = agg["year"].tolist()
    y_hours = agg["finish_h"].tolist()
    txt = [_seconds_to_hms(v) for v in agg["finish_sec"].tolist()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_years,
            y=y_hours,
            width=0.2,
            marker_color="#4A4A4A", 
            text=txt,
            textposition="outside",
            hovertemplate="<b>Year</b>: %{x}<br><b>Median finish</b>: %{text}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_years,
            y=y_hours,
            marker_color="orange", 
            mode="lines+markers",
            hovertemplate="<b>Year</b>: %{x}<br><b>Median finish</b>: %{text}<extra></extra>",
            text=txt,
        )
    )

    fig.update_layout(
        paper_bgcolor="#7a7a7a",
        plot_bgcolor="#7a7a7a",
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            title="Year",
            type="category",
            categoryorder="array",
            categoryarray=x_years,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Median finish time (hours)",
            showgrid=True,
            zeroline=False,
            tickformat=".1f",
        ),
        showlegend=False,
    )

    return fig


# ------------------------------------------------------------
# Public render function (used by Overview page)
# ------------------------------------------------------------
def render_year_comparison(selected_year, selected_group) -> None:
    
    title_col, info_col = st.columns([0.94, 0.06], vertical_alignment="center")
    with title_col:
        st.subheader("Year comparison: Median leg times till 32.5 km")
    with info_col:
        with st.popover("ℹ️"):
            st.markdown(
        """
**What is shown in this chart?**

This visualization compares **median segment times by year** for the Norseman race up to **32.5 km on the run**.
Each year contains **three horizontal bars**:

- **Swim**: median `swim_time`  
- **Bike**: median `bike_time`  
- **Run (to 32.5 km)**: median (`run_32_5km_langefonn_time - run_start_time`)  

**Longer bar = slower** (values are shown in hours on the x-axis).

**How to interpret this chart**

- Compare **within a year** to see which segment contributes most to time  
- Compare **across years** to spot performance shifts (faster/slower medians)  
- In **Compared to 2025** mode:
  - **Green** = faster than 2025 median  
  - **Red** = slower than 2025 median  
  - **Grey** = 2025 baseline / no change  

**How this can be used**

- Track long-term changes in segment performance year-over-year  
- Identify which segment drives changes in overall performance across editions  
- Benchmark a selected finisher group (Top 10 / Black / White / DNF) consistently across years  
- Support training focus decisions by seeing where time is most “expensive”


        """
    )

    df = _load_wide_df()

    # ignore selected_year on purpose (always compare all years)
    df = _apply_group_filter(df, selected_group)

    show_in = st.radio(
        "Display values as",
        ["Time Overview", "Compared to 2025", "Trend"],
        horizontal=True,
        index=0,
        key="year_compare_units",
    )

    if df.empty:
        st.info("No data available for the selected group.")
        return

    med = _compute_year_medians(df)
    if med.empty:
        st.info("No median values available (missing splits for this selection).")
        return

    if show_in == "Trend":
        fig2 = _build_finish_trend_chart(df)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        fig = _build_year_compare_chart(med, show_in=show_in)
        st.plotly_chart(fig, use_container_width=True)

    piv = (
        med.pivot_table(index="year", columns="leg", values="median_sec", aggfunc="first")
        .reset_index()
        .sort_values("year")
    )

    # Ensure year is displayed as normal integer (2018 not 2,018 / not float)
    piv["year"] = pd.to_numeric(piv["year"], errors="coerce").astype("Int64").astype(int)

    def fmt(v):
        return _seconds_to_hms(v) if show_in == "Time Overview" else _seconds_to_min_str(v)

    piv_disp = piv.copy()
    for col in piv_disp.columns:
        if col != "year":
            piv_disp[col] = piv_disp[col].apply(fmt)


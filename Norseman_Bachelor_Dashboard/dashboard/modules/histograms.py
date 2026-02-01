import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np


GRAY_BG = "#7A7A7A"
TOP10_COLOR = "lightgreen"

TIME_COLUMN_CANDIDATES = {
    "swim": ["swim_time", "SwimTime", "Swim"],
    "bike": ["bike_time", "BikeTime", "Bike"],
    "run": ["run_time", "RunTime", "Run"],
    "finish": ["overall_time", "FinishTime", "Total_Time", "Total", "Finish"],
}

YEAR_CANDIDATES = ["year", "Year"]
FINISH_TYPE_CANDIDATES = ["finish_type", "FinishType", "Finisher_Status"]
OVERALL_RANK_CANDIDATES = ["overall_rank", "Overall_Rank"]
PARTICIPANT_ID_CANDIDATES = ["AthleteId", "athlete_id", "athleteId", "BIB", "bib"]

SPLIT_KEY_CANDIDATES = ["split_key", "SplitKey"]
SPLIT_TIME_CANDIDATES = ["split_time", "SplitTime", "time", "Time", "split_time_sec", "split_seconds"]

RUN_START_CANDIDATES = ["Run start", "Run Start", "RunStart", "run_start"]

CUTOFF_KEY_37_5 = "run_37_5km_stavsro_cut_off"
CUTOFF_KEY_32_5 = "run_32_5km_langefonn"

CUTOFF_YEARS_32_5 = {2018, 2019, 2021, 2022}


# Helpers

def detect_column(df, candidates):
    """Ersten existierenden Spaltennamen aus candidates zurückgeben."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def format_hms(seconds: float) -> str:
    """Sekunden -> 'HH:MM:SS'."""
    if pd.isna(seconds):
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_seconds(series: pd.Series) -> pd.Series:
    """timedelta / hh:mm:ss / Minuten / Sekunden => Sekunden (float)."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()

    if pd.api.types.is_numeric_dtype(series):
        s = series.astype(float)
        # Heuristik: <1000 => Minuten
        if s.max() < 1000:
            return s * 60.0
        return s

    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()


def pick_label(unique_vals, preferred_order):
    for val in preferred_order:
        if val in unique_vals:
            return val
    return None


def build_color_map(unique_vals):
    """Farbmapping je nach tatsächlichen Finish-Type-Werten im DF."""
    color_map = {}
    black = pick_label(unique_vals, ["Black Shirt", "Black"])
    if black:
        color_map[black] = "black"
    white = pick_label(unique_vals, ["White Shirt", "White"])
    if white:
        color_map[white] = "white"
    dnf = pick_label(unique_vals, ["DNF", "Dnf", "dnf"])
    if dnf:
        color_map[dnf] = "red"
    return color_map


def build_rounded_ticks(sec_min, sec_max, min_step_minutes=5):
    """Runde Ticks in hh:mm:ss (5–30-Minuten Raster)."""
    if not (math.isfinite(sec_min) and math.isfinite(sec_max)):
        return None, None
    if sec_max <= sec_min:
        return None, None

    min_m = sec_min / 60.0
    max_m = sec_max / 60.0
    span = max_m - min_m

    if span > 240:
        step = 30
    elif span > 120:
        step = 20
    elif span > 60:
        step = 10
    else:
        step = 5

    step = max(step, min_step_minutes)

    start = math.floor(min_m / step) * step
    end = math.ceil(max_m / step) * step

    tick_mins = []
    v = start
    while v <= end + 1e-9:
        tick_mins.append(v)
        v += step

    tickvals = [m * 60.0 for m in tick_mins]
    ticktext = [format_hms(s) for s in tickvals]
    return tickvals, ticktext


# Filtering Helpers

def apply_header_filters(df, selected_year, selected_group,
                         year_col, finish_type_col, overall_rank_col, unique_vals):

    filtered = df.copy()

    # Year-Filter
    if year_col is not None and selected_year != "All":
        filtered = filtered[filtered[year_col].astype(str) == str(selected_year)]

    # Group "Top 10"
    if selected_group == "Top 10" and overall_rank_col is not None:
        filtered = filtered[filtered[overall_rank_col] <= 10]

    # Group nach Finish Type (Black/White/DNF)
    if finish_type_col is not None:
        if selected_group == "Black Shirt":
            label = pick_label(unique_vals, ["Black Shirt", "Black"])
            if label:
                filtered = filtered[filtered[finish_type_col] == label]

        elif selected_group == "White Shirt":
            label = pick_label(unique_vals, ["White Shirt", "White"])
            if label:
                filtered = filtered[filtered[finish_type_col] == label]

        elif selected_group == "DNF":
            label = pick_label(unique_vals, ["DNF", "Dnf", "dnf"])
            if label:
                filtered = filtered[filtered[finish_type_col] == label]

    return filtered


# Plot Helper 

def create_histogram(
    df, time_col, title,
    finish_type_col, overall_rank_col, color_map,
    scale_mode="Absolute (hh:mm:ss)",
    reference_range=None,
):
    fig = go.Figure()
    df = df[df[time_col].notna()].copy()

    if df.empty:
        fig.update_layout(
            title=dict(text=title, x=0.01, xanchor="left"),
            paper_bgcolor=GRAY_BG,
            plot_bgcolor=GRAY_BG,
            font=dict(color="black"),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    df["_seconds"] = to_seconds(df[time_col])

    if scale_mode == "Relative (% vs median)":
        med = df["_seconds"].median()
        med = med if (med is not None and med > 0 and math.isfinite(med)) else None
        df["_x"] = ((df["_seconds"] - med) / med) if med else df["_seconds"]
        x_axis_title = "Relative time vs median (%)"

    elif scale_mode == "Bike-Time scale (min swim → 10h)":
        df["_x"] = df["_seconds"]
        x_axis_title = "Time (hh:mm:ss) — Bike-Time scale"

    else:
        df["_x"] = df["_seconds"]
        x_axis_title = "Time (hh:mm:ss)"

    # Legend order: Top 10 | Black | White | DNF
    legend_rank_map = {
        "Top 10": 0,
        "Black": 1,
        "Black Shirt": 1,
        "White": 2,
        "White Shirt": 2,
        "DNF": 3,
        "Dnf": 3,
        "dnf": 3,
    }

    # Top 10 exklusiv: aus den Basis-Gruppen rausnehmen
    top10_ids = set()
    if overall_rank_col is not None and overall_rank_col in df.columns:
        top10_ids = set(df.loc[df[overall_rank_col] <= 10].index)

    df_base = df.drop(index=top10_ids, errors="ignore")
    df_top10 = df.loc[list(top10_ids)].copy() if len(top10_ids) > 0 else df.iloc[0:0].copy()

    # globale Edges für Relative / Bike-Time-scale
    global_edges = None

    # Relative: global edges für Vergleichbarkeit
    if scale_mode == "Relative (% vs median)":
        gx = df["_x"].dropna().astype(float)
        if not gx.empty:
            gmin = float(gx.min())
            gmax = float(gx.max())
            if math.isfinite(gmin) and math.isfinite(gmax) and gmax > gmin:
                global_edges = np.linspace(gmin, gmax, 41)  # 40 Bins

    # Bike-Time-scale: alle Disziplinen nutzen dieselben Edges wie Referenz
    if scale_mode == "Bike-Time scale (min swim → 10h)" and reference_range is not None:
        rmin, rmax = reference_range
        if math.isfinite(rmin) and math.isfinite(rmax) and rmax > rmin:
            global_edges = np.linspace(rmin, rmax, 41)  # 40 Bins

    def add_binned_bar(sub_df, name, color, nbinsx, opacity, legendrank):
        if sub_df.empty:
            return

        x = sub_df["_x"].dropna().astype(float)
        if x.empty:
            return

        x_min = float(x.min())
        x_max = float(x.max())
        if not (math.isfinite(x_min) and math.isfinite(x_max)) or x_max <= x_min:
            return

        # Edges: Relative/RaceDay -> global, Absolute -> per group
        if scale_mode in ["Relative (% vs median)", "Bike-Time scale (min swim → 10h)"] and global_edges is not None:
            edges = global_edges
        else:
            edges = np.linspace(x_min, x_max, nbinsx + 1)

        centers = (edges[:-1] + edges[1:]) / 2.0
        width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0

        counts, _ = np.histogram(x.values, bins=edges)

        hovertext = []
        for i, c in enumerate(counts):
            a = edges[i]
            b = edges[i + 1]
            if scale_mode in ["Absolute (hh:mm:ss)", "Bike-Time scale (min swim → 10h)"]:
                hovertext.append(f"{name}<br>{format_hms(a)} – {format_hms(b)}<br>n={int(c)}")
            else:
                hovertext.append(f"{name}<br>{(a*100):.0f}% – {(b*100):.0f}%<br>n={int(c)}")

        marker = dict(color=color)
        if str(color).lower() == "white":
            marker["line"] = dict(color="#888888", width=1)

        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=width,
                name=name,
                opacity=opacity,
                marker=marker,
                legendrank=legendrank,
                hovertext=hovertext,
                hoverinfo="text",
            )
        )

    # Black / White / DNF (OHNE Top 10)
    for ft, color in color_map.items():
        sub = df_base[df_base[finish_type_col] == ft]
        is_dnf = str(ft).lower() == "dnf"
        add_binned_bar(
            sub_df=sub,
            name=ft,
            color=color,
            nbinsx=80 if is_dnf else 40,
            opacity=0.65 if is_dnf else 0.7,
            legendrank=legend_rank_map.get(ft, 10),
        )

    # Top 10 (exklusiv)
    if not df_top10.empty:
        add_binned_bar(
            sub_df=df_top10,
            name="Top 10",
            color=TOP10_COLOR,
            nbinsx=40,
            opacity=0.79,
            legendrank=0,
        )

    # Ticks nur im absoluten / Bike-Time-scale Modus (hh:mm:ss)
    tickvals, ticktext = None, None
    if scale_mode in ["Absolute (hh:mm:ss)", "Bike-Time scale (min swim → 10h)"]:
        if scale_mode == "Bike-Time scale (min swim → 10h)" and reference_range is not None:
            sec_min, sec_max = reference_range
        else:
            sec_min = df["_seconds"].min()
            sec_max = df["_seconds"].max()
        tickvals, ticktext = build_rounded_ticks(float(sec_min), float(sec_max))

    fig.update_layout(
        title=dict(text=title, y=0.92, x=0.01, xanchor="left"),
        barmode="overlay",
        paper_bgcolor=GRAY_BG,
        plot_bgcolor=GRAY_BG,
        font=dict(color="black"),
        margin=dict(l=10, r=10, t=90, b=10),
        legend=dict(
            y=1.22,
            x=0.99,
            xanchor="right",
            yanchor="top",
        ),
    )

    if scale_mode in ["Absolute (hh:mm:ss)", "Bike-Time scale (min swim → 10h)"]:
        fig.update_xaxes(
            title=dict(text=x_axis_title, font=dict(size=14,color="white")),
            tickfont=dict(color="white", size=12),
            gridcolor="#c7c7c7",
            tickmode="array" if tickvals is not None else "auto",
            tickvals=tickvals,
            ticktext=ticktext,
            zerolinecolor="#c7c7c7",
            range=list(reference_range) if (scale_mode == "Bike-Time scale (min swim → 10h)" and reference_range is not None) else None,
        )
    else:
        fig.update_xaxes(
            title=dict(text=x_axis_title, font=dict(size=14,color="white")),
            tickfont=dict(size=12,color="white"),
            gridcolor="#c7c7c7",
            zerolinecolor="#c7c7c7",
            tickformat=".0%",
        )

    fig.update_yaxes(
        title=dict(text="Frequency", font=dict(size=14,color="white")),
        tickfont=dict(color="white", size=12),
        gridcolor="#c7c7c7",
        zerolinecolor="#c7c7c7",
    )

    # Median-Linie für Relative-Skala (0 %)
    if scale_mode == "Relative (% vs median)":
        fig.add_vline(
            x=0,
            line=dict(
                color="orange",
                width=2,
                dash="dot",
            ),
            annotation_text="Median",
            annotation_position="top",
            annotation_font=dict(
                color="orange",
                size=12,
            ),
        )


    return fig



# Haupt-Render-Funktion 

def render(df: pd.DataFrame, selected_year, selected_group):

    year_col = detect_column(df, YEAR_CANDIDATES)
    finish_type_col = detect_column(df, FINISH_TYPE_CANDIDATES)
    overall_rank_col = detect_column(df, OVERALL_RANK_CANDIDATES)

    swim_col = detect_column(df, TIME_COLUMN_CANDIDATES["swim"])
    bike_col = detect_column(df, TIME_COLUMN_CANDIDATES["bike"])
    run_col = detect_column(df, TIME_COLUMN_CANDIDATES["run"])
    finish_col = detect_column(df, TIME_COLUMN_CANDIDATES["finish"])

    if finish_type_col is None:
        st.warning("No finish type column found.")
        return

    # Finish-Type-Werte fürs Farb-Mapping
    unique_vals = set(map(str, df[finish_type_col].dropna().unique()))
    color_map = build_color_map(unique_vals)

    # Titel + Toggle + Info rechts
   # Zeile 1: Überschrift links + Info rechts
    title_col, info_col = st.columns([0.92, 0.08], vertical_alignment="center")

    with title_col:
        st.markdown("### Time distributions")

    # Zeile 2: Radio Buttons direkt unter der Überschrift
    scale_mode = st.radio(
        "Scale",
        ["Absolute (hh:mm:ss)", "Relative (% vs median)", "Bike-Time scale (min swim → 10h)"],
        horizontal=True,
        label_visibility="collapsed",
    )


    with info_col:
        with st.popover("ℹ️"):
            st.markdown("""
 **What is shown in these charts?**

These histograms show the **distribution of leg and total times**

- **X-axis**: time per race segment  
  - *Absolute*: time in **hh:mm:ss**  
  - *Relative*: time shown as **% slower/faster than the median** of that segment  
  - *Bike-Time scale*: time in **hh:mm:ss**, but swim/bike/run use a **common axis**
    from the **fastest swim** to **10 hours** (≈ slowest bike)
- **Y-axis**: frequency (number of athletes per time bin)


 **How to interpret these charts**
- A distribution that is **shifted left** represents **faster athletes**
- A **wider spread** indicates greater performance variability
- Overlapping bars show **similar performance ranges** between groups
- In *Relative* mode, differences between swim, bike, run, and finish
  become **directly comparable**
- In *Bike-Time scale*, the magnitude of leg times becomes visually comparable
  (swim/run appear compressed relative to bike)

 **How this can be used**
- Identify **which race segment creates the largest separation**
- Compare **Top 10 vs. Black / White finishers** across all legs
- Support strategic insights on **pacing and decisive race sections**

Athletes are grouped by finish category (**Top 10, Black, White, DNF**).
*Note: DNF athletes times are shown only when they finished the respective segment.*
""")


    # Year + Group Filter anwenden
    df_filtered = apply_header_filters(
        df, selected_year, selected_group,
        year_col, finish_type_col, overall_rank_col, unique_vals
    )

    # Nur eine Zeile pro Athlet
    pid_col = detect_column(df_filtered, PARTICIPANT_ID_CANDIDATES)
    if pid_col is not None:
        df_plot = df_filtered.drop_duplicates(subset=[pid_col]).copy()
    else:
        df_plot = df_filtered.copy()

    # Referenzbereich für gemeinsame Achse: min Swim -> max(10h, max Bike)
    reference_range = None
    if scale_mode == "Bike-Time scale (min swim → 10h)":
        swim_min = None
        if swim_col is not None:
            s = to_seconds(df_plot[swim_col]).dropna()
            if not s.empty:
                swim_min = float(s.min())

        bike_max = None
        if bike_col is not None:
            b = to_seconds(df_plot[bike_col]).dropna()
            if not b.empty:
                bike_max = float(b.max())

        upper = max(10 * 3600.0, bike_max) if bike_max is not None else 10 * 3600.0

        if swim_min is not None and math.isfinite(swim_min) and swim_min > 0 and upper > swim_min:
            pad = (upper - swim_min) * 0.02
            reference_range = (max(0.0, swim_min - pad), upper + pad)

    # Run/Finish bis Cutoff-Distanz (year-aware, auch wenn selected_year=="All") — Long-only via split_key
    split_key_col = detect_column(df_filtered, SPLIT_KEY_CANDIDATES)
    split_time_col = detect_column(df_filtered, SPLIT_TIME_CANDIDATES)
    pid_col_long = detect_column(df_filtered, PARTICIPANT_ID_CANDIDATES)

    cutoff_station_col = None
    run_start_col = None

    if split_key_col is not None and split_time_col is not None and pid_col_long is not None and year_col is not None:
        idx_cols = [pid_col_long, year_col]

        # wir ziehen beide cutoff keys + run_start, dann wählen wir pro Jahr
        wanted_keys = [CUTOFF_KEY_32_5, CUTOFF_KEY_37_5, "run_start"]
        df_keys = df_filtered[df_filtered[split_key_col].isin(wanted_keys)].copy()

        lookup = (
            df_keys[idx_cols + [split_key_col, split_time_col]]
            .pivot_table(index=idx_cols, columns=split_key_col, values=split_time_col, aggfunc="first")
            .reset_index()
        )

        # df_plot ist schon 1 Zeile pro Athlet -> merge passt
        df_plot = df_plot.merge(lookup, on=idx_cols, how="left")

        run_start_col = "run_start" if "run_start" in df_plot.columns else None

        # Jahr entscheiden: 32.5 oder 37.5
        y = pd.to_numeric(df_plot[year_col], errors="coerce")
        use_32 = y.isin(list(CUTOFF_YEARS_32_5))

        has_32 = CUTOFF_KEY_32_5 in df_plot.columns
        has_37 = CUTOFF_KEY_37_5 in df_plot.columns

        if has_32 or has_37:
            df_plot["_total_to_cutoff"] = pd.NA

            if has_32 and has_37:
                df_plot.loc[use_32, "_total_to_cutoff"] = df_plot.loc[use_32, CUTOFF_KEY_32_5]
                df_plot.loc[~use_32, "_total_to_cutoff"] = df_plot.loc[~use_32, CUTOFF_KEY_37_5]
            elif has_32:
                df_plot["_total_to_cutoff"] = df_plot[CUTOFF_KEY_32_5]
            else:
                df_plot["_total_to_cutoff"] = df_plot[CUTOFF_KEY_37_5]

            cutoff_station_col = "_total_to_cutoff"

            if run_start_col is not None:
                cutoff_sec = to_seconds(df_plot["_total_to_cutoff"])
                runstart_sec = to_seconds(df_plot[run_start_col])
                df_plot["_run_to_cutoff_sec"] = cutoff_sec - runstart_sec
            else:
                df_plot["_run_to_cutoff_sec"] = pd.NA

    # Dynamische Titel für Run/Total (Cutoff)
    run_title = "Run Time"
    finish_title = "Finish Time"

    if cutoff_station_col is not None:
        run_title = "Run Time (to cut-off)"
        finish_title = "Time to cut-off"

    # Vier Plots nebeneinander
    col1, col2, col3, col4 = st.columns(4)

    if swim_col is not None:
        with col1:
            st.plotly_chart(
                create_histogram(
                    df_plot, swim_col, "Swim Time",
                    finish_type_col, overall_rank_col, color_map,
                    scale_mode=scale_mode,
                    reference_range=reference_range,
                ),
                width="stretch",
            )

    if bike_col is not None:
        with col2:
            st.plotly_chart(
                create_histogram(
                    df_plot, bike_col, "Bike Time",
                    finish_type_col, overall_rank_col, color_map,
                    scale_mode=scale_mode,
                    reference_range=reference_range,
                ),
                width="stretch",
            )

    with col3:
        if cutoff_station_col is not None and "_run_to_cutoff_sec" in df_plot.columns:
            st.plotly_chart(
                create_histogram(
                    df_plot, "_run_to_cutoff_sec", run_title,
                    finish_type_col, overall_rank_col, color_map,
                    scale_mode=scale_mode,
                    reference_range=reference_range,
                ),
                width="stretch",
            )
        elif run_col is not None:
            st.plotly_chart(
                create_histogram(
                    df_plot, run_col, "Run Time (till Cut-Off)",
                    finish_type_col, overall_rank_col, color_map,
                    scale_mode=scale_mode,
                    reference_range=reference_range,
                ),
                width="stretch",
            )
        else:
            st.info("No run time column found.")

    with col4:
        if cutoff_station_col is not None and "_total_to_cutoff" in df_plot.columns:
            st.plotly_chart(
                create_histogram(
                    df_plot, "_total_to_cutoff", finish_title,
                    finish_type_col, overall_rank_col, color_map,
                    scale_mode=scale_mode,
                ),
                width="stretch",
            )
        elif finish_col is not None:
            st.plotly_chart(
                create_histogram(
                    df_plot, finish_col, "Time to Cut-Off",
                    finish_type_col, overall_rank_col, color_map,
                    scale_mode=scale_mode,
                ),
                width="stretch",
            )
        else:
            st.info("No finish time column found.")

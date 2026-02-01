# timerelations.py

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# Konstanten
# --------------------------------------------------

SWIM_DISTANCE_KM = 3.8
BIKE_DISTANCE_KM = 180.2
RUN_DISTANCE_KM = 42.2

SEGMENT_LABELS = ["Swim", "Bike", "Run"]
SEGMENT_DISTANCES = [SWIM_DISTANCE_KM, BIKE_DISTANCE_KM, RUN_DISTANCE_KM]

SEGMENT_COLORS = {
    "Swim": "#64b5f6",   
    "Bike": "#4A4A4A",   
    "Run": "#2e7d32",    
}

BACKGROUND_DARK = "#7A7A7A"
TEXT_COLOR = "#ffffff"

# Farben für Überschriften
TOP10_BG = "#b7ffb7"
TOP10_FONT = "#000000"

BLACK_BG = "#000000"
BLACK_FONT = "#ffffff"

WHITE_BG = "#ffffff"
WHITE_FONT = "#000000"


# --------------------------------------------------
# Helper-Funktionen
# --------------------------------------------------

def time_to_seconds(t) -> float:
    """Konvertiert 'h:mm:ss' oder 'mm:ss' in Sekunden."""
    if pd.isna(t) or t is None or t == "":
        return np.nan
    t = str(t).strip()
    parts = t.split(":")

    try:
        if len(parts) == 2:          # mm:ss
            h = 0
            m, s = parts
        elif len(parts) == 3:        # h:mm:ss
            h, m, s = parts
        else:
            return np.nan
        return int(h) * 3600 + int(m) * 60 + int(round(float(s)))
    except Exception:
        return np.nan


def seconds_to_hms(sec: float) -> str:
    """Formatiert Sekunden als h:mm:ss (leer, wenn NaN)."""
    if sec is None or np.isnan(sec):
        return ""
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}"


def format_delta(delta_sec: float) -> str:
    """Formatiert Delta-Sekunden als 'Δ +h:mm:ss' oder 'Δ -h:mm:ss' (0 immer +00:00:00)."""
    if delta_sec is None or np.isnan(delta_sec):
        return ""
    if int(round(delta_sec)) == 0:
        return "Δ +00:00:00"
    sign = "+" if delta_sec > 0 else "-"
    return f"Δ {sign}{seconds_to_hms(abs(delta_sec))}"


@st.cache_data
def load_wide_for_timerelations() -> pd.DataFrame:
    """
    Lädt die gleiche CSV wie die Pacetabelle und erzeugt:
    - swim_time_s, bike_time_s, run_time_s
    - Top10_flag
    """
    df = pd.read_csv(
        "/Users/manuelmiethke/Norseman_Bachelor/old_data/old_master_data_excel/nxtri_data_all_years.csv"
    )

    df["swim_time_s"] = df["swim_time"].apply(time_to_seconds)
    df["bike_time_s"] = df["bike_time"].apply(time_to_seconds)
    df["run_time_s"] = df["run_time"].apply(time_to_seconds)
    df["Top10_flag"] = df["overall_rank"].le(10)

    return df


def _compute_time_share_for_group(
    df: pd.DataFrame,
    group: str,
    ref_black: Optional[dict] = None,
    delta_only: bool = False,
) -> Optional[dict]:
    """
    Berechnet Zeitanteile (in %) und Text für Swim/Bike/Run für:
    - 'Top 10'
    - 'Black Shirt'
    - 'White Shirt'

    Anzeige:
      - delta_only = False -> nur Zeit + Prozent
      - delta_only = True  -> nur Delta (Black = Δ +00:00:00)
    """
    # Nur Finisher (keine DNF)
    df = df[df["finish_type"].isin(["Black", "White"])]

    if group == "Top 10":
        df_group = df[df["Top10_flag"]]
    elif group == "Black Shirt":
        df_group = df[df["finish_type"] == "Black"]
    elif group == "White Shirt":
        df_group = df[df["finish_type"] == "White"]
    else:
        return None

    if df_group.empty:
        return None

    # Median statt Mean
    swim_sec = df_group["swim_time_s"].median()
    bike_sec = df_group["bike_time_s"].median()
    run_sec = df_group["run_time_s"].median()

    values = [swim_sec, bike_sec, run_sec]
    if any(np.isnan(v) for v in values):
        return None

    total = sum(values)
    if not total or math.isnan(total):
        return None

    pct = [v / total * 100 for v in values]

    # Delta vs Black (Median)
    if not ref_black:
        deltas = [np.nan, np.nan, np.nan]
    else:
        deltas = [
            swim_sec - ref_black["swim"],
            bike_sec - ref_black["bike"],
            run_sec - ref_black["run"],
        ]

    if delta_only:
        if group == "Black Shirt":
            text = ["Δ +00:00:00", "Δ +00:00:00", "Δ +00:00:00"]
        else:
            text = [format_delta(d) for d in deltas]
    else:
        text = [
            f"{seconds_to_hms(v)}<br>{p:.1f} %"
            for v, p in zip(values, pct)
        ]

    return {"pct": pct, "text": text}


def _create_pie(
    title: str,
    values: List[float],
    labels: List[str],
    text: List[str],
    title_bg: str | None = None,
    title_font_color: str | None = None,
) -> go.Figure:
    """
    Erzeugt ein einzelnes Pie-Chart mit Dark-Theme
    und fetter Titel-Box. Alle Zahlen stehen im Donut,
    waagerecht und in einheitlicher Größe.
    """
    colors = [SEGMENT_COLORS[l] for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=colors),
                text=text,
                textinfo="text",
                textposition="inside",
                insidetextorientation="horizontal",
                textfont=dict(size=18, color=TEXT_COLOR),
                hovertemplate="%{label}<br>%{text}<extra></extra>",
            )
        ]
    )

    annotations = [
        dict(
            text=f"<b>{title}</b>",
            x=0.5,
            y=1.18,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                color=title_font_color or TEXT_COLOR,
                size=24,
                family="sans-serif",
            ),
            bgcolor=title_bg if title_bg is not None else "rgba(0,0,0,0)",
            align="center",
            borderpad=6,
        )
    ]

    fig.update_layout(
        paper_bgcolor=BACKGROUND_DARK,
        plot_bgcolor=BACKGROUND_DARK,
        font=dict(color=TEXT_COLOR),
        margin=dict(t=90, b=20, l=10, r=10),
        showlegend=False,
        annotations=annotations,
        uniformtext_minsize=18,
        uniformtext_mode="show",
        height=360,
    )

    return fig


# --------------------------------------------------
# Hauptfunktion
# --------------------------------------------------

def render_time_relations(df: pd.DataFrame | None = None) -> None:
    """
    Zeigt bis zu 4 Kuchendiagramme nebeneinander:
    - Distance
    - je nach group_filter: Top 10 / Black Shirt / White Shirt / alle

    Checkbox: nur Delta anzeigen (sonst nur Time + %)
    """
    needed_cols = {"swim_time_s", "bike_time_s", "run_time_s", "finish_type"}
    if df is None or not needed_cols.issubset(df.columns):
        df = load_wide_for_timerelations()

    selected_year = st.session_state.get("year_filter", "All")
    if selected_year != "All" and "year" in df.columns:
        df = df[df["year"] == selected_year]

    selected_group = st.session_state.get("group_filter", "All")

    # -----------------------------
    # Header + Info-Icon Box
    # -----------------------------
    title_col, info_col = st.columns([0.94, 0.06], vertical_alignment="center")
    with title_col:
        st.markdown("### Distance vs. Time Relations")
    with info_col:
        with st.popover("ℹ️"):
            st.markdown(
                """
**What is shown in this chart?**

This visualization compares **distance distribution** with **time distribution**
across the three Norseman race segments (*Swim, Bike, Run*), split by athlete groups
(**Top 10**, **Black Shirt**, **White Shirt**).

- **Distance** shows how the total race distance is distributed across the three segments  
- The other charts show the **median time share** spent in each segment  
- All Delta time-based charts use **Black Shirt (median)** as the reference group  

**How to interpret this chart**

- Differences between **distance share** and **time share** highlight which segments
  dominate overall race time  
- A higher time share than distance share indicates a **performance-critical segment**  
- Comparing groups shows how performance levels differ between **Top 10**, **Black**, and **White** finishers  

**How this can be used**

- Understand which race segments matter most for overall performance  
- Benchmark group performance against the **Black Shirt median**  
- Identify where time gains or losses typically occur across athlete groups  
- Support pacing and training focus decisions based on segment importance


                """
            )

    delta_only = st.checkbox("Show Δ compared to Black Shirts", value=False)

    # ----------------------------------------------------------
    # Distance-Kuchen
    # ----------------------------------------------------------
    total_distance = sum(SEGMENT_DISTANCES)
    distance_pct = [d / total_distance * 100 for d in SEGMENT_DISTANCES]

    distance_text = [
        f"{dist:.1f} km<br>{pct:.1f} %"
        for dist, pct in zip(SEGMENT_DISTANCES, distance_pct)
    ]

    fig_distance = _create_pie(
        "Distance",
        distance_pct,
        SEGMENT_LABELS,
        distance_text,
        title_bg=None,
        title_font_color=TEXT_COLOR,
    )

    # ----------------------------------------------------------
    # Black-Referenz (Median) für Deltas
    # ----------------------------------------------------------
    df_fin = df[df["finish_type"].isin(["Black", "White"])]
    df_black = df_fin[df_fin["finish_type"] == "Black"]

    ref_black = None
    if not df_black.empty:
        ref_black = {
            "swim": df_black["swim_time_s"].median(),
            "bike": df_black["bike_time_s"].median(),
            "run": df_black["run_time_s"].median(),
        }

    # ----------------------------------------------------------
    # Zeit-Kuchen nach Gruppen
    # ----------------------------------------------------------
    charts: List[go.Figure] = [fig_distance]

    def add_group_chart(group_name: str, bg: str, font: str):
        result = _compute_time_share_for_group(
            df,
            group_name,
            ref_black=ref_black,
            delta_only=delta_only,
        )
        if result is not None:
            fig = _create_pie(
                group_name,
                result["pct"],
                SEGMENT_LABELS,
                result["text"],
                title_bg=bg,
                title_font_color=font,
            )
            charts.append(fig)

    if selected_group == "All":
        add_group_chart("Top 10", TOP10_BG, TOP10_FONT)
        add_group_chart("Black Shirt", BLACK_BG, BLACK_FONT)
        add_group_chart("White Shirt", WHITE_BG, WHITE_FONT)
    elif selected_group == "Top 10":
        add_group_chart("Top 10", TOP10_BG, TOP10_FONT)
    elif selected_group == "Black Shirt":
        add_group_chart("Black Shirt", BLACK_BG, BLACK_FONT)
    elif selected_group == "White Shirt":
        add_group_chart("White Shirt", WHITE_BG, WHITE_FONT)
    elif selected_group == "DNF":
        pass

    cols = st.columns(len(charts))
    for col, fig in zip(cols, charts):
        with col:
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False},
            )

    if selected_group == "DNF":
        st.info(
            "Für DNF-Athlet:innen wird keine Zeitaufteilung angezeigt, "
            "da sie den Triathlon nicht komplett absolviert haben."
        )


# Debug-View
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_time_relations()

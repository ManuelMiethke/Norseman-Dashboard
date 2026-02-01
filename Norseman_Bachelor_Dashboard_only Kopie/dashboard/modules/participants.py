import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --------------------------------------------------
# Flag Mapping (so wie davor)
# --------------------------------------------------
FLAG_MAP = {
    "NOR": "🇳🇴", "NO": "🇳🇴",
    "GBR": "🇬🇧", "GB": "🇬🇧", "UNK": "🇬🇧", "UK": "🇬🇧",
    "FRA": "🇫🇷", "FR": "🇫🇷",
    "USA": "🇺🇸", "US": "🇺🇸",
    "SWE": "🇸🇪",
    "GER": "🇩🇪", "DE": "🇩🇪",
    "ITA": "🇮🇹",
    "NED": "🇳🇱", "NLD": "🇳🇱",
    "CAN": "🇨🇦",
    "POL": "🇵🇱", "PL": "🇵🇱",
    "SUI": "🇨🇭",
    "FIN": "🇫🇮",
    "EST": "🇪🇪",
    "AUT": "🇦🇹",
    "BEL": "🇧🇪",
}

# --------------------------------------------------
# Country normalization (nur fürs Zählen & Deduplizieren)
# --------------------------------------------------
COUNTRY_NORMALIZE = {
    "POL": "PL",   
    "DEU": "DE",
    "GER": "DE",
    "GBR": "GB",
    "UK": "GB",
    "ENG": "GB",
    "USA": "US",
    "NOR": "NO",
}

def normalize_country(code: str) -> str:
    if pd.isna(code):
        return "UNK"
    c = str(code).strip().upper()
    return COUNTRY_NORMALIZE.get(c, c)


def get_flag(code: str) -> str:
    """
    Mapping wie davor:
    1) erst exakt wie geliefert (damit "wie vorher")
    2) falls nicht gefunden: normalisierte Variante probieren
    """
    if not isinstance(code, str):
        return "🏳️"
    raw = code.strip()
    if raw in FLAG_MAP:
        return FLAG_MAP.get(raw, "🏳️")
    norm = normalize_country(raw)
    return FLAG_MAP.get(norm, "🏳️")


# --------------------------------------------------
# Statistik-Berechnung
# --------------------------------------------------
def compute_stats(df: pd.DataFrame, selected_year):
    """
    Berechnet:
    - Gesamtstarter
    - Finish-Typen (Counts & %)
    - Gender (Counts & %)
    - Top-6 Länder (+ Rest-%)
    - Anzahl teilnehmender Länder (dedupliziert, ohne UNK)
    """
    participants_df = df.drop_duplicates(["year", "bib"]).copy()
    years = sorted(participants_df["year"].unique())

    def stats_single_year(y: int):
        sub = participants_df[participants_df["year"] == y]
        if sub.empty:
            return None

        # Starterzahl
        n_starters = int(sub["n_starters_year"].max())

        # Finish-Typen
        black = sub[sub["finish_type"] == "Black"]["bib"].nunique()
        white = sub[sub["finish_type"] == "White"]["bib"].nunique()
        dnf_known = sub[sub["finish_type"] == "DNF"]["bib"].nunique()

        # DNF-Regel
        if y in (2019, 2025):
            dnf = dnf_known
        else:
            dnf = max(n_starters - black - white, 0)

        # Gender
        men = sub[sub["gender"] == "M"]["bib"].nunique()
        women = sub[sub["gender"] == "F"]["bib"].nunique()
        total_gender = men + women or 1

        # Länder (normalisiert fürs Zählen, damit POL/PL zusammen)
        country_series = sub["country"].apply(normalize_country)
        country_counts = country_series.value_counts()
        total_country = country_counts.sum() or 1

        # Participating countries (ohne UNK)
        n_countries = int(country_series[country_series != "UNK"].nunique())

        topN_stats = []
        for c, cnt in country_counts.head(6).items():
            pct = cnt / total_country * 100
            topN_stats.append(
                {"code": c, "flag": get_flag(c), "count": int(cnt), "pct": pct}
            )
        rest_pct = max(0.0, 100 - sum(x["pct"] for x in topN_stats))

        def pct(n, base):
            return (n / base * 100) if base else 0.0

        return {
            "n_starters": n_starters,
            "finish_counts": {"black": black, "white": white, "dnf": dnf},
            "finish_pct": {
                "black": pct(black, n_starters),
                "white": pct(white, n_starters),
                "dnf": pct(dnf, n_starters),
            },
            "gender_counts": {"men": men, "women": women},
            "gender_pct": {
                "men": pct(men, total_gender),
                "women": pct(women, total_gender),
            },
            "country_stats": topN_stats,
            "country_rest_pct": rest_pct,
            "n_countries": n_countries,
        }

    # All Years aggregiert 
    if selected_year == "All":
        per_year_stats = []
        for y in years:
            s = stats_single_year(int(y))
            if s is not None:
                per_year_stats.append(s)

        if not per_year_stats:
            return None

        n_starters = sum(s["n_starters"] for s in per_year_stats)
        black = sum(s["finish_counts"]["black"] for s in per_year_stats)
        white = sum(s["finish_counts"]["white"] for s in per_year_stats)
        dnf = sum(s["finish_counts"]["dnf"] for s in per_year_stats)

        men = sum(s["gender_counts"]["men"] for s in per_year_stats)
        women = sum(s["gender_counts"]["women"] for s in per_year_stats)
        total_gender = men + women or 1

        # Länder für All (normalisiert)
        country_series_all = participants_df["country"].apply(normalize_country)
        cc = country_series_all.value_counts()
        total_country = cc.sum() or 1

        n_countries_all = int(country_series_all[country_series_all != "UNK"].nunique())

        topN_stats = []
        for c, cnt in cc.head(6).items():
            pct = cnt / total_country * 100
            topN_stats.append(
                {"code": c, "flag": get_flag(c), "count": int(cnt), "pct": pct}
            )
        rest_pct = max(0.0, 100 - sum(x["pct"] for x in topN_stats))

        def pct(n, base):
            return (n / base * 100) if base else 0.0

        return {
            "n_starters": n_starters,
            "finish_counts": {"black": black, "white": white, "dnf": dnf},
            "finish_pct": {
                "black": pct(black, n_starters),
                "white": pct(white, n_starters),
                "dnf": pct(dnf, n_starters),
            },
            "gender_counts": {"men": men, "women": women},
            "gender_pct": {
                "men": pct(men, total_gender),
                "women": pct(women, total_gender),
            },
            "country_stats": topN_stats,
            "country_rest_pct": rest_pct,
            "n_countries": n_countries_all,
        }

    # Einzeljahr
    return stats_single_year(int(selected_year))


# --------------------------------------------------
# Plotly: gestackte Balken 
# --------------------------------------------------
def stacked_bar_plot(labels, values, colors, height=28):
    fig = go.Figure()

    for label, val, col in zip(labels, values, colors):
        fig.add_trace(
            go.Bar(
                x=[val],
                y=[""],
                name=label,
                orientation="h",
                marker=dict(color=col),
                hovertemplate=f"{label}: %{{x}}<extra></extra>",
            )
        )

    total = sum(values) if sum(values) > 0 else 1

    fig.update_traces(marker_line_width=0)

    fig.update_layout(
        barmode="stack",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, total], fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# --------------------------------------------------
# Card-Helfer 
# --------------------------------------------------
def base_card_html(title: str, inner_html: str) -> str:
    return f"""
    <div style="
        background-color:#7A7A7A;
        padding:18px;
        margin-top:16px;
        margin-bottom:16px;
        font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color:#000000;
    ">
            <style>
            div, span, p, strong {{
                color: #ffffff !important;
                font-size: 1.1rem;
            }}
        </style>
        <h4 style="margin-top:0; margin-bottom:6px; font-size:1.1rem;">
            {title}
        </h4>

        {inner_html}
    </div>
    """


def render_total_participants_card(total_participants: int, n_countries: int = None, height: int = 96):
    right = f"""
        <div style="font-size:2.2rem; font-weight:700; line-height:1; color:#ffffff;">{n_countries:,}</div>
        <div style="font-size:0.85rem; opacity:0.85; color:#ffffff;">Participating countries</div>
    """ if n_countries is not None else ""

    card_html = f"""
    <div style="
        background-color:#7A7A7A;
        padding:16px;
        margin-top:16px;
        margin-bottom:16px;
        color:#ffffff;
        font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
    <style>
        div, span, p, strong {{
            color: #ffffff !important;
        }}
    </style>
            <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="flex:1; text-align:center;">
                <div style="font-size:2.2rem; font-weight:700; line-height:1; color:#ffffff;">{total_participants:,}</div>
                <div style="font-size:0.85rem; opacity:0.85; color:#ffffff;">Total participants</div>
            </div>

            <div style="width:1px; height:48px; background:rgba(255,255,255,0.35); margin:0 14px;"></div>

            <div style="flex:1; text-align:center;">
                {right}
            </div>
        </div>
    </div>
    """
    components.html(card_html, height=height, scrolling=False)


def render_gender_card(gender_counts, gender_pct, height: int = 140):
    men_count = gender_counts["men"]
    women_count = gender_counts["women"]
    men_pct = gender_pct["men"]
    women_pct = gender_pct["women"]

    fig = stacked_bar_plot(
        ["Men", "Women"],
        [men_count, women_count],
        colors=["#3b82f6", "#f973c5"],
        height=28,
    )

    fig_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    inner_html = f"""
        {fig_html}
        <div style="
            display:flex;
            justify-content:space-between;
            margin-top:6px;
            font-size:0.95rem;
            color:#ffffff;
        ">
            <div style="color:#ffffff;">Men: <strong>{men_count}</strong> ({men_pct:.1f}%)</div>
            <div style="text-align:right; color:#ffffff;">
                Women: <strong>{women_count}</strong> ({women_pct:.1f}%)
            </div>
        </div>
    """

    card_html = base_card_html("Gender", inner_html)
    components.html(card_html, height=height, scrolling=False)


def render_finish_types_card(finish_counts, finish_pct, height: int = 150):
    black_c = finish_counts["black"]
    white_c = finish_counts["white"]
    dnf_c = finish_counts["dnf"]

    black_p = finish_pct["black"]
    white_p = finish_pct["white"]
    dnf_p = finish_pct["dnf"]

    fig = stacked_bar_plot(
        ["Black", "White", "DNF"],
        [black_c, white_c, dnf_c],
        colors=["#000000", "#ffffff", "#ef4444"],
        height=28,
    )

    fig_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    inner_html = f"""
        {fig_html}
        <div style="
            display:flex;
            width:100%;
            margin-top:6px;
            font-size:0.95rem;
            color:#ffffff;
        ">
            <div style="flex:0 0 {black_p:.1f}%; text-align:left; color:#ffffff;">
                Black: <strong>{black_c}</strong> ({black_p:.1f}%)
            </div>
            <div style="flex:0 0 {white_p:.1f}%; text-align:center; color:#ffffff;">
                White: <strong>{white_c}</strong> ({white_p:.1f}%)
            </div>
            <div style="flex:0 0 {dnf_p:.1f}%; text-align:right; color:#ffffff;">
                DNF: <strong>{dnf_c}</strong> ({dnf_p:.1f}%)
            </div>
        </div>
    """

    card_html = base_card_html("Finish types", inner_html)
    components.html(card_html, height=height, scrolling=False)


def render_countries_card(country_stats, rest_pct, height: int = 165):
    items_html = ""
    for cinfo in country_stats:
        items_html += f"""
        <div style="
            flex:1 1 0;
            text-align:center;
            color:#ffffff;
        ">
            <div style="font-size:3.0rem; margin-bottom:2px; line-height:1;">{cinfo['flag']}</div>
            <div style="font-weight:600; margin-bottom:0px; color:#ffffff;">{cinfo['pct']:.1f}%</div>
            <div style="font-size:0.8rem; opacity:0.85; color:#ffffff;">{cinfo['code']}</div>
        </div>
        """

    items_html += f"""
    <div style="
        flex:1 1 0;
        text-align:center;
        color:#ffffff;
    ">
        <div style="font-size:3.0rem; margin-bottom:2px; line-height:1;">➕</div>
        <div style="font-weight:600; margin-bottom:0px; color:#ffffff;">{rest_pct:.1f}%</div>
        <div style="font-size:0.8rem; opacity:0.85; color:#ffffff;">Rest</div>
    </div>
    """

    inner_html = f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            width:100%;
            margin-top:4px;
        ">
            {items_html}
        </div>
    """

    card_html = base_card_html("Most participants", inner_html)
    components.html(card_html, height=height, scrolling=False)


# --------------------------------------------------
# Country participation over time (Stacked Area)
# --------------------------------------------------
def compute_country_time_series(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.drop_duplicates(["year", "bib"]).copy()
        .assign(country_norm=lambda d: d["country"].apply(normalize_country))
    )

    counts = (
        base.groupby(["year", "country_norm"])
        .size()
        .reset_index(name="count")
    )
    return counts


def top_n_with_rest(df_counts: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    top_countries = (
        df_counts.groupby("country_norm")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )

    df_counts = df_counts.copy()
    df_counts["country_plot"] = df_counts["country_norm"].where(
        df_counts["country_norm"].isin(top_countries),
        "OTHER"
    )

    out = (
        df_counts.groupby(["year", "country_plot"])["count"]
        .sum()
        .reset_index()
        .sort_values(["year", "country_plot"])
    )
    return out


def _label_for_country(code: str) -> str:
    if code == "OTHER":
        return "Other"
    if code == "DE":
        return "GER"
    return str(code)


def _color_for_country(code: str) -> str:
    # feste, undurchsichtige Farben (Alpha = 1)
    # (Plotly: rgba(r,g,b,1) = 100% opacity)
    PALETTE = {
    # 🇳🇴 Norway → Rot (Flagge)
    "NO": "rgba(220,38,38,1)",

    # 🇫🇷 France → Dunkelblau (Flagge)
    "FRA": "rgba(42,90,152,1)",

    # 🇸🇪 Sweden → Gelb (Flagge)
    "SWE": "rgba(234,179,8,1)",

    # 🇳🇱 Netherlands → Orange (Flagge, leicht dunkler als GB)
    "NL": "rgba(234,88,12,1)",

    # 🇬🇧 UK → Navy / Royal Blue (Flagge, sehr UK-typisch)
    "GB": "rgba(99,102,241,1)",

    # 🇩🇪 Germany → Grün (Abhebung)
    "DE": "rgba(128,141,35,1)",
    # 🇺🇸 USA → Crimson / Dark Red-Blue Mix (klassisch)
    "US": "rgba(153,27,27,1)",

    # Rest
    "OTHER": "rgba(150,150,150,1)",
}
    return PALETTE.get(code, "rgba(150,150,150,1)")

def render_country_trend_area(df: pd.DataFrame, top_n: int = 6):
    df_counts = compute_country_time_series(df)
    plot_df = top_n_with_rest(df_counts, n=top_n)

    years = sorted(plot_df["year"].unique())

    # Reihenfolge so, dass die größten Flächen UNTEN liegen:
    totals = (
        plot_df.groupby("country_plot")["count"]
        .sum()
        .sort_values(ascending=False)
    )
    ordered_countries = totals.index.tolist()  # first trace = bottom

    # Label soll irgendwo "im Block" auftauchen:
    mid_idx = len(years) // 2
    label_year = years[mid_idx] if years else None

    fig = go.Figure()

    for c in ordered_countries:
        sub = plot_df[plot_df["country_plot"] == c].set_index("year").reindex(years).reset_index()
        sub["count"] = sub["count"].fillna(0)

        # Text nur an einem Punkt (damit es nicht spammt)
        text = [""] * len(sub)
        if label_year is not None:
            # Index des Mid-Year in sub (sollte existieren, da reindex)
            i = sub.index[sub["year"] == label_year]
            if len(i) > 0:
                text[int(i[0])] = _label_for_country(c)

        fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["count"],
                mode="lines+text",
                stackgroup="one",
                name=_label_for_country(c),
                fillcolor=_color_for_country(c),
                line=dict(width=0),
                text=text,
                textposition="middle center",
                textfont=dict(size=14, color="white"),
                hovertemplate="<b>%{x}</b><br>%{y:.0f} participants<extra></extra>",
                showlegend=False,  # keine Legende
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=None, tickmode="array", tickvals=years),
        yaxis=dict(title="Participants"),
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# UI Render Function (gleich wie davor, nur erweitert)
# --------------------------------------------------
def render_participants(df: pd.DataFrame, selected_year):
    stats = compute_stats(df, selected_year)
    if stats is None:
        st.write("No data available for this selection.")
        return

    render_total_participants_card(stats["n_starters"], stats.get("n_countries"), height=96)

    render_gender_card(stats["gender_counts"], stats["gender_pct"], height=140)
    render_finish_types_card(stats["finish_counts"], stats["finish_pct"], height=150)
    render_countries_card(stats["country_stats"], stats["country_rest_pct"], height=165)

    with st.expander("Participation by country over time", expanded=False):
        render_country_trend_area(df, top_n=6)

import streamlit as st
from pathlib import Path
from utils.race_logic import GROUP_CRITICAL_40

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logos" / "black-shirt_icon.jpg"

def render_header():
    # Optionen
    year_options = ["All", 2025, 2024, 2022, 2021, 2019, 2018]
    group_options = ["All", "Top 10", "Black Shirt", "White Shirt", GROUP_CRITICAL_40, "DNF"]

    col_title, col_year, col_group, col_logo = st.columns([4.2, 1.3, 1.5, 1])

    with col_title:
        st.markdown(
            "<h1 style='margin-bottom:0.2rem;'>Norseman Black-Shirt Dashboard</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            """
<p style='margin:0; color:"white"; font-size:1.0rem;'>
  Explore performances at the Norseman Xtreme Triathlon
  · <a href='https://my.raceresult.com/321377/' target='_blank'>Data</a>
  · (Best viewed on a large display)
</p>
            """,
            unsafe_allow_html=True,
        )

    with col_year:
        st.markdown('<p class="filter-label">Year</p>', unsafe_allow_html=True)
        selected_year = st.selectbox(
            "Year",
            year_options,
            index=1,
            key="year_filter",
            label_visibility="collapsed"
        )

    with col_group:
        st.markdown('<p class="filter-label">Group</p>', unsafe_allow_html=True)
        selected_group = st.selectbox(
            "Group",
            group_options,
            index=0,
            key="group_filter",
            label_visibility="collapsed"
        )

    with col_logo:
        st.image(str(LOGO_PATH), width=110)

    return selected_year, selected_group

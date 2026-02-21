import streamlit as st
from pathlib import Path
from utils.race_logic import GROUP_CRITICAL_40

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logos" / "black-shirt_icon.jpg"

def render_header():
    # Optionen
    year_options = ["All", 2025, 2024, 2022, 2021, 2019, 2018]
    group_options = ["All", "Top 10", "Black Shirt", "White Shirt", GROUP_CRITICAL_40, "DNF"]

    col_title, col_source, col_year, col_group, col_logo = st.columns([3.7, 0.5, 1.3, 1.5, 1])

    with col_title:
        st.markdown(
            "<h1 style='margin-bottom:0.2rem;'>Norseman Black-Shirt Dashboard</h1>",
            unsafe_allow_html=True
        )
        st.caption("Explore performances at the Norseman Xtreme Triathlon")

    with col_source:
        with st.popover("📊"):
            st.markdown(
                """
**Data sources (RaceResults):**

- [2018](https://my.raceresult.com/100840/)
- [2019](https://my.raceresult.com/108938/)
- [2021](https://my.raceresult.com/164078/results)
- [2022](https://my.raceresult.com/210752/results)
- [2024](https://my.raceresult.com/298522/results)
- [2025](https://my.raceresult.com/321377/)
                """
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

# layout.py
import os
import streamlit as st

from header import render_header
import data_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(BASE_DIR, "assets", "styles.css")


def apply_global_style():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
            background-color: #000000 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
        }

        /* Text default (statt *-Override) */
        body, p, div, span, label, h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
        }

        /* ----------------------------
           NAVBAR: hide "main"
           ---------------------------- */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="main"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="/main"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="?page=main"] {
            display: none !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)




def setup_page(page_title: str = "Norseman Dashboard"):
    st.set_page_config(page_title=page_title, layout="wide")
    apply_global_style()

    df_long = data_store.df_long()
    selected_year, selected_group = render_header()

    return df_long, selected_year, selected_group

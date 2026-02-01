import streamlit as st
from PIL import Image

NORSEMAN_LOGO_PATH = "/Users/manuelmiethke/Norseman_Bachelor_Dashboard_only/dashboard/assets/logos/black-shirt_icon.jpg"
HDM_LOGO_PATH = "/Users/manuelmiethke/Norseman_Bachelor/dashboard/assets/logos/HdM_logo.png"


def footer() -> None:
    # dünne Trennlinie oben
    st.markdown("---")

    # Spaltenlayout: Text links/mittig, Logos rechts
    col_left, col_center, col_contact, col_ns_logo, col_hdm_logo = st.columns(
        [1.2, 2, 1.2, 0.8, 0.8]
    )

    # Links: Norseman-Link
    with col_left:
        st.markdown("[Norseman](https://www.nxtri.com/)")

    # Mitte: Last updated
    with col_center:
        st.markdown("Last updated: **21.12.2025**")

    # Rechts: Contact (Mailto-Link)
    with col_contact:
        st.markdown("[Contact](mailto:mm326@hdm-stuttgart.de)")

    # Norseman-Logo
    try:
        ns_logo = Image.open(NORSEMAN_LOGO_PATH)
        with col_ns_logo:
            st.image(ns_logo, width=70)
    except Exception:
        with col_ns_logo:
            st.write("")

    # HdM-Logo
    try:
        hdm_logo = Image.open(HDM_LOGO_PATH)
        with col_hdm_logo:
            st.image(hdm_logo, width=70)
    except Exception:
        with col_hdm_logo:
            st.write("")

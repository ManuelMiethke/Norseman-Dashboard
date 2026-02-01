from pathlib import Path
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent

ASSETS_DIR = DASHBOARD_DIR / "assets"

MAIN_DATA_PATH = ASSETS_DIR / "nxtri_data_all_years_long_ready_featured.csv"
COURSE_PROFILE_PATH = ASSETS_DIR / "course_profile.csv"
MODEL_DATA_PATH = ASSETS_DIR / "xgb_black_probs_long_with_errors.csv"
WIDE_DATA_PATH = ASSETS_DIR / "nxtri_data_all_years.csv"


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")

# ------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def df_long(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else MAIN_DATA_PATH
    _assert_exists(p, "Main long CSV")
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def course_profile(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else COURSE_PROFILE_PATH
    _assert_exists(p, "Course profile CSV")
    df = pd.read_csv(p)
    if "distance_km" in df.columns:
        df = df.sort_values("distance_km")
    return df


@st.cache_data(show_spinner=False)
def df_model(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else MODEL_DATA_PATH
    _assert_exists(p, "Model CSV")
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def df_wide(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else WIDE_DATA_PATH
    _assert_exists(p, "Wide CSV")
    return pd.read_csv(p)
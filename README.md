# Norseman Black-Shirt Dashboard (Streamlit)

An interactive **Streamlit dashboard** for analyzing performance data from the **Norseman Xtreme Triathlon (NXTRI)**, with a particular focus on pacing strategies, elevation effects, predictive modeling, and Black-Shirt qualification.

---

## Features

### Overview
- Participant and result overviews  
- Time and split relationships  
- Histograms and ranking progressions  
- Year-to-year comparisons  

### Pacing & Elevation
- Weather vs. speed analysis  
- Cumulative times and segment breakdowns  
- Pace tables and boxplots  
- Pacing heatmap along the race course  

### Prediction
- Model performance evaluation (accuracy and error analysis)  
- **Black-Shirt probability** along the race distance  

### Tools
- Empirical pace calculators (e.g. target finish time 13:51)  
- “Catch-up” scenarios and overtaking calculators  

---

## Project Structure

```text
Norseman_Bachelor_Dashboard_only/
└─ dashboard/
   ├─ pages/                    # Streamlit multi-page applications
   │  ├─ 01_Overview.py
   │  ├─ 02_Pacing_&_Course.py
   │  ├─ 03_Model Predictions.py
   │  └─ 04_Tools.py
   ├─ modules/                  # Reusable visualization and logic modules
   │  ├─ participants.py
   │  ├─ timerelations.py
   │  ├─ histograms.py
   │  ├─ rangverlauf_*.py
   │  ├─ weatherspeed.py
   │  ├─ blackshirt_prob.py
   │  ├─ modelaccuracy.py
   │  ├─ what_are_the_odds.py
   │  └─ empirical_pace.py
   ├─ assets/                   # Data and static assets
   │  ├─ nxtri_data_all_years.csv
   │  ├─ nxtri_data_all_years_long_ready_featured.csv
   │  ├─ xgb_black_probs_long_with_errors.csv
   │  ├─ course_profile.csv
   │  └─ logos/
   ├─ data_store.py             # Central data loading and caching layer (st.cache_data)
   ├─ layout.py                 # Page setup and global styles
   ├─ header.py                 # Global filters (year / group) and header
   ├─ footer.py                 # Footer, logos, and links
   └─ main.py                   # Entry point (currently empty)
```

---

## Requirements

- Python **3.10+**
- Required packages:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `plotly`
  - `Pillow`

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -U pip
pip install streamlit pandas numpy plotly pillow
```

Optional: generate a `requirements.txt` file

```bash
pip freeze > requirements.txt
```

---

## Running the App

The project is implemented as a **Streamlit multi-page application** using `dashboard/pages/`.

```bash
streamlit run dashboard/main.py
```

**Note:**  
`dashboard/main.py` is currently empty.  
Navigation is handled via the Streamlit sidebar.  
A dedicated landing page can be added later if required.

---

## Data

CSV files are loaded from `dashboard/assets/` via `dashboard/data_store.py` using `st.cache_data`:

- `nxtri_data_all_years_long_ready_featured.csv` – main dataset (long format)  
- `nxtri_data_all_years.csv` – alternative / wide format  
- `xgb_black_probs_long_with_errors.csv` – model output (Black-Shirt probabilities)  
- `course_profile.csv` – course elevation profile  

---

## Development Notes

- Add new pages by creating files in `dashboard/pages/`  
  (file numbering controls the sidebar order).
- Place reusable visualizations and logic in `dashboard/modules/`.
- All shared data access should go through `dashboard/data_store.py`.

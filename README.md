Norseman Black-Shirt Dashboard (Streamlit)

Interactive Streamlit dashboard for analyzing performance data from the Norseman Xtreme Triathlon (NXTRI) – including overview, pacing/elevation, model predictions, and tools.
Features
Overview
Participant and result overviews
Time / split relationships
Histograms & ranking progressions
Year-to-year comparisons
Pacing & Elevation
Weather vs. speed
Cumulative times / segments
Pace tables & boxplots
Pacing heatmap along the course
Prediction
Model quality (accuracy / error analysis)
Black-Shirt probability along the race distance
Tools
Empirical pace calculators (e.g. target time 13:51)
“Catch-up” scenarios / overtaking calculators


Norseman_Bachelor_Dashboard_only/
└─ dashboard/
   ├─ pages/                    # Streamlit Multi‑Page Apps Pages
   │  ├─ 01_Overview.py
   │  ├─ 02_Pacing_&_Course.py
   │  ├─ 03_Model Predictions.py
   │  └─ 04_Tools.py
   ├─ modules/                  # reusable Diagram modules
   │  ├─ participants.py
   │  ├─ timerelations.py
   │  ├─ histograms.py
   │  ├─ rangverlauf_*.py
   │  ├─ weatherspeed.py
   │  ├─ blackshirt_prob.py
   │  ├─ modelaccuracy.py
   │  ├─ catchup.py
   │  └─ empirical_pace.py
   ├─ assets/                   # Data and static assets
   │  ├─ nxtri_data_all_years.csv
   │  ├─ nxtri_data_all_years_long_ready_featured.csv
   │  ├─ xgb_black_probs_long_with_errors.csv
   │  ├─ course_profile.csv
   │  └─ logos/
   ├─ data_store.py             # Data (st.cache_data)
   ├─ layout.py                 # Page-Setup + Global Styles
   ├─ header.py                 # Header
   ├─ footer.py                 # Footer
   └─ main.py                   # empty right now–  Entry‑Point for Streamlit
   
Requirements:
Python 3.10+
Dependencies (minimum):
streamlit
pandas
numpy
plotly
Pillow
Installation

1) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell

3) Install dependencies
pip install -U pip
pip install streamlit pandas numpy plotly pillow

Running the App
The project is structured as a Streamlit multi-page app (dashboard/pages/).
Start (using dashboard/main.py as entry point)
streamlit run dashboard/main.py

Important:
Currently, dashboard/main.py is empty.
Use the sidebar page list;
main.py still needs to be populated as a landing page.

Data
The app loads CSV files from dashboard/assets/ via dashboard/data_store.py (using st.cache_data):
nxtri_data_all_years_long_ready_featured.csv – long format (main dataset)
nxtri_data_all_years.csv – wide / alternative format
xgb_black_probs_long_with_errors.csv – model output (Black-Shirt probabilities)
course_profile.csv – course profile

Development
New page: create a file in dashboard/pages/ (numbering controls order).
Reusable visuals / logic: place in dashboard/modules/.
Shared data access: via dashboard/data_store.py.

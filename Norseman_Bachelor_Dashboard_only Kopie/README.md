# Norseman Black‑Shirt Dashboard (Streamlit)

Interaktives **Streamlit-Dashboard** zur Analyse von Leistungsdaten des **Norseman Xtreme Triathlon (NXTRI)** – inkl. Overview, Pacing/Elevation, Modell‑Vorhersagen  und Tools.

---

## Features

- **Overview**
  - Teilnehmer- und Ergebnis-Überblicke
  - Zeit-/Split-Relationen
  - Histogramme & Rangverläufe
  - Jahresvergleiche
- **Pacing & Elevation**
  - Wetter vs. Speed
  - Akkumulierte Zeiten/Segmente
  - Pace-Tabellen & Boxplots
  - Pacing-Heatmap entlang der Strecke
- **Prediction**
  - Modellgüte (Accuracy / Fehleranalysen)
  - **Black‑Shirt‑Probability** entlang der Renndistanz
- **Tools**
  - Empirische Pace-Rechner (z. B. Zielzeit 13:51)
  - „Catch‑Up“-Szenarien / Aufholrechner

---

## Projektstruktur

```text
Norseman_Bachelor_Dashboard_only/
└─ dashboard/
   ├─ pages/                    # Streamlit Multi‑Page Apps (Seiten)
   │  ├─ 01_Overview.py
   │  ├─ 02_Pacing_&_Course.py
   │  ├─ 03_Model Predictions.py
   │  └─ 04_Tools.py
   ├─ modules/                  # Wiederverwendbare Visualisierungs-/Logikmodule
   │  ├─ participants.py
   │  ├─ timerelations.py
   │  ├─ histograms.py
   │  ├─ rangverlauf_*.py
   │  ├─ weatherspeed.py
   │  ├─ blackshirt_prob.py
   │  ├─ modelaccuracy.py
   │  ├─ catchup.py
   │  └─ empirical_pace.py
   ├─ assets/                   # Daten & statische Assets
   │  ├─ nxtri_data_all_years.csv
   │  ├─ nxtri_data_all_years_long_ready_featured.csv
   │  ├─ xgb_black_probs_long_with_errors.csv
   │  ├─ course_profile.csv
   │  └─ logos/
   ├─ data_store.py             # Zentrale Datenlade-/Cache-Schicht (st.cache_data)
   ├─ layout.py                 # Page-Setup + Global Styles
   ├─ header.py                 # Filter (Year/Group) + Header
   ├─ footer.py                 # Footer/Logos/Links
   └─ main.py                   # (derzeit leer) – typischer Entry‑Point für Streamlit
```

---

## Voraussetzungen

- Python **3.10+** 
- Abhängigkeiten (mindestens):
  - `streamlit`
  - `pandas`
  - `numpy`
  - `plotly`
  - `Pillow`

---

## Installation

### 1) Virtuelle Umgebung anlegen 

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

### 2) Dependencies installieren

```bash
pip install -U pip
pip install streamlit pandas numpy plotly pillow
```

Optional: Erzeuge ein `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## App starten

Das Projekt ist als **Streamlit Multi‑Page App** aufgebaut (`dashboard/pages/`).

###  Start ( `dashboard/main.py` als Entry‑Point genutzt)

```bash
streamlit run dashboard/main.py
```

> **Wichtig:** In der aktuell ist `dashboard/main.py` leer.  
> Seitenliste in der Sidebar nutzen,
 `main.py` als Landing‑Page sollte noch befüllt werden 

---

## Daten

Die App lädt CSVs aus `dashboard/assets/` über `dashboard/data_store.py` (mit `st.cache_data`):

- `nxtri_data_all_years_long_ready_featured.csv` – Long-Format (Hauptdaten)
- `nxtri_data_all_years.csv` – Wide/Alternative
- `xgb_black_probs_long_with_errors.csv` – Modell-Output (Black‑Shirt‑Probs)
- `course_profile.csv` – Streckenprofil

---

## Entwicklung

- Neue Seite: Datei in `dashboard/pages/` anlegen (Nummerierung steuert Reihenfolge).
- Wiederverwendbare Visuals/Logik: in `dashboard/modules/`.
- Gemeinsame Datenzugriffe: über `dashboard/data_store.py`.

---

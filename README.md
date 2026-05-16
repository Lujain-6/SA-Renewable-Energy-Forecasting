# Renewable Energy Forecasting - Saudi Arabia

Predicting Renewable Energy Capacity in Saudi Arabia using Machine Learning and Data Analysis.

## Overview

This project analyzes renewable energy growth in Saudi Arabia using official datasets from Data Saudi and KAPSARC.

The system processes installed and planned renewable energy projects, generates visualizations, and forecasts future renewable energy capacity using a Linear Regression model.

The project focuses on:
- Renewable energy growth analysis
- Solar vs Wind comparison
- Regional distribution analysis
- Forecasting renewable energy capacity until 2030
- Comparing forecasted results with Vision 2030 targets

---

## Live Interactive MLOps Production Dashboard

### Key Dashboard Features:
- **Dynamic User Filtering:** Users can interactively select specific Saudi cities and target years via the sidebar control panel.
- **Live Reactive KPIs:** Total tracked projects, cumulative capacity, and average project dimensions dynamically recalculate and update in real-time based on the user's selection.
- **Global Predictive Models:** To preserve machine learning model integrity, statistical dimensions, and prevent data density errors (such as single-row array crashes), the primary ML regression plots and country-wide forecasting trends remain unified to show the complete Vision 2030 national trajectory.
- **Automated Alignment Report:** Dynamic parsing of $R^2$ scores, Vision 2030 target achievement rates, and gap analyses directly from the predictive pipeline.

---

## CI/CD Automation (MLOps Pipeline)

This repository implements **GitHub Actions** for Continuous Integration and Continuous Deployment (CI/CD). Every code modification automatically triggers a secure, automated cloud runner that:
1. Validates code linting and structural configurations.
2. Installs required environment specifications and libraries from `requirements.txt`.
3. Tests the core backend execution (`main.py`).
4. Compiles and checks the production dashboard (`app.py`) for syntax integrity before staging live.

---

## Forecasting Model

The project uses a Linear Regression model to learn historical cumulative renewable energy growth trends and forecast future renewable energy capacity until 2030.

The forecasted results are compared with the Vision 2030 renewable energy benchmark of approximately 58.7 GW.

---

## Code Structure

| File / Function | Description |
|---|---|
| `load_data()` | Load renewable energy datasets |
| `clean_data1(data1)` | Clean the first dataset |
| `clean_data2(data2)` | Clean and standardize the second dataset |
| `create_raw_dataset(data1, data2)` | Merge cleaned datasets |
| `train_renewable_model(X, y)` | Train Linear Regression forecasting model |
| `plot_yearly_growth(df_raw)` | Visualize yearly renewable energy growth |
| `plot_solar_vs_wind(df_raw)` | Compare solar and wind energy |
| `plot_regional_distribution(df_raw)` | Visualize regional distribution |
| `plot_forecast(df_raw)` | Forecast renewable energy capacity |
| `evaluate_forecast(...)` | Evaluate forecasting performance |
| `app.py` | Core Streamlit interactive user-centric interface |

## Project Pipeline

Data Collection  
→ Data Preprocessing  
→ Forecasting Model  
→ Forecast Evaluation  
→ Visualization  
→ CI/CD Automation (GitHub Actions)  
→ Dashboard Deployment (Streamlit)

## Evaluation Metrics

- Trend Fit (R²)
- Achievement Rate
- Gap Analysis
- Estimated Yearly Growth

## Getting Started

Install required packages:

pip install -r requirements.txt

Place the following datasets inside the `data/` folder:

- saudi-arabia-planned-and-installed-renewables-by-project.csv
- renewable_energy_projects.csv

To spin up the web dashboard locally, run:

streamlit run app.py

## Tech Stack

- **Backend:** Python, pandas, NumPy, Scikit-learn (Linear Regression)
- **Frontend & Deployment:** Streamlit, Matplotlib
- **Automation:** GitHub Actions (YAML CI/CD Workflows)

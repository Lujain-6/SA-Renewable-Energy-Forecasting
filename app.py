import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1️⃣ Import real functions from your project files
from data_loader import load_data
from preprocessing import clean_data1, clean_data2

# Resolve merge function import location dynamically
try:
    from preprocessing import merge
except ImportError:
    from model import merge

from visualization import plot_forecast
from evaluation import evaluate_forecast

# Application page configuration
st.set_page_config(page_title="Saudi Renewable Energy AI", page_icon="🇸🇦", layout="wide")

st.title("🇸🇦 Saudi Arabia Renewable Energy Forecasting Platform")
st.subheader("Automated MLOps Production Dashboard Linked directly to main.py scripts")

st.write("---")

# Sidebar dashboard control panel
st.sidebar.header("🛠️ Pipeline Controller")
run_pipeline = st.sidebar.button("🚀 Run Live AI Pipeline")

# Execute core pipeline upon button click or default load
if run_pipeline or True:
    with st.spinner("Executing core main.py pipeline models..."):
        try:
            # 2️⃣ Execute core main.py pipeline steps sequentially
            data1_raw, data2_raw = load_data()
            d1_cleaned = clean_data1(data1_raw)
            d2_cleaned = clean_data2(data2_raw)
            df_raw = merge(d1_cleaned, d2_cleaned)
            
            st.success("🎯 Backend Pipeline executed successfully from main.py files!")
            
            # 3️⃣ Extract live forecast values from your plot_forecast function for Year 2030
            future_2030, vision_target, trend_fit, yearly, slope = plot_forecast(df_raw, forecast_until=2030)
            
            # 4️⃣ Display real system KPI metrics calculated by your ML model
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Calculated Slope (Growth Rate)", value=f"{slope:.2f} MW/Year")
            with col2:
                st.metric(label="Predicted Capacity by 2030", value=f"{int(future_2030):,} MW")
            with col3:
                st.metric(label="Saudi Vision 2030 Target", value=f"{int(vision_target):,} MW")
                
            st.write("---")
            
            # 5️⃣ Display merged dataset preview and generated charts side-by-side
            left_col, right_col = st.columns(2)
            with left_col:
                st.subheader("📋 Core Cleaned & Merged Data")
                st.dataframe(df_raw.head(20), use_container_width=True)
                
            with right_col:
                st.subheader("📈 Generated AI System Forecast")
                # Render the generated chart image if saved by the pipeline script
                if os.path.exists("forecast_chart.png"): 
                    st.image("forecast_chart.png", caption="Vision 2030 Live System Forecast")
                else:
                    # Fallback live plotting block if image file is not found
                    fig, ax = plt.subplots()
                    ax.plot(df_raw['Year'], df_raw['Cumulative Capacity'], marker='o', color='green')
                    ax.set_title("Historical Cleaned Data Growth")
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Pipeline Execution Error: {str(e)}")
            st.info("Make sure all raw dataset files are correctly placed in the 'data/' folder in GitHub.")

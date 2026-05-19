import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the exact functions from your team's files
from data_loader import load_data
from preprocessing import clean_data1, clean_data2, merge
from visualization import plot_yearly_growth, plot_solar_vs_wind, plot_regional_distribution, plot_forecast_by_status
from evaluation import evaluate_dual_forecasts

# Application page configuration
st.set_page_config(page_title="Saudi Renewable Energy AI", layout="wide")

st.title("Saudi Arabia Renewable Energy Forecasting Platform")
st.subheader("An interactive AI-driven MLOps dashboard analyzing and forecasting Saudi Arabia's Vision 2030 renewable energy targets.")

st.write("---")

# Sidebar dashboard control panel
st.sidebar.header("🛠️ Pipeline Controller")
run_pipeline = st.sidebar.button("🚀 Run Live AI Pipeline")

# Execute core pipeline upon button click or default load
if run_pipeline or True:
    with st.spinner("Executing core main.py pipeline models..."):
        try:
            # 2️⃣ Load and clean the raw data using team's functions
            data1_raw, data2_raw = load_data()
            d1_cleaned = clean_data1(data1_raw)
            d2_cleaned = clean_data2(data2_raw)
            df_merged = merge(d1_cleaned, d2_cleaned)
            
            # 🛠️ CRITICAL COLUMNS MATCHING FIX
            # Enforce 'Capacity' column synchronization to satisfy custom team plotting functions
            if 'Capacity (MW)' in df_merged.columns:
                df_merged['Capacity'] = df_merged['Capacity (MW)']
            if 'Capacity (MW)' in d1_cleaned.columns:
                d1_cleaned['Capacity'] = d1_cleaned['Capacity (MW)']

            st.success("🎯 Backend Pipeline executed successfully from main.py files!")
            
            # 3️⃣ Calculate metrics dynamically from the actual data
            total_projects = len(df_merged)
            total_current_capacity = df_merged['Capacity'].sum()
            avg_capacity = df_merged['Capacity'].mean()
            
            # 4️⃣ Display real system KPI metrics calculated from your actual data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Tracked Projects", value=f"{total_projects} Projects")
            with col2:
                st.metric(label="Total System Capacity", value=f"{int(total_current_capacity):,} MW")
            with col3:
                st.metric(label="Average Project Capacity", value=f"{avg_capacity:.2f} MW")
                
            st.write("---")
            
            # 5️⃣ Grid Layout for Visualizations & User-Centric Explanations
            st.subheader("📊 Team System Visualizations & AI Forecasts")
            
            # First row of charts
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Yearly Renewable Growth (Installed)")
                fig1 = plt.figure(figsize=(8, 4.5))
                plot_yearly_growth(d1_cleaned)
                st.pyplot(plt.gcf())
                plt.close(fig1)
                # 💡 User-Centric Explanation
                st.info("**AI Insight:** This chart illustrates the historical trend of established renewable installations. The green bars track annual capacity additions, while the dark line tracks cumulative infrastructure growth across Saudi Arabia.")
                
            with c2:
                st.write("### Solar vs Wind Capacity Comparison")
                fig2 = plt.figure(figsize=(8, 4.5))
                plot_solar_vs_wind(df_merged)
                st.pyplot(plt.gcf())
                plt.close(fig2)
                # 💡 User-Centric Explanation
                st.info("**AI Insight:** This breakdown compares the total capacity distribution between Solar and Wind projects. It helps energy professionals assess the diversification and balance of the Kingdom's current renewable energy mix.")
                
            # Second row of charts
            c3, c4 = st.columns(2)
            with c3:
                st.write("### Regional Project Distribution")
                fig3 = plt.figure(figsize=(8, 4.5))
                plot_regional_distribution(df_merged)
                st.pyplot(plt.gcf())
                plt.close(fig3)
                # 💡 User-Centric Explanation
                st.info("**AI Insight:** This geographic summary identifies which Saudi provinces host the largest renewable setups, pinpointing major investment hubs like Tabuk (NEOM) and Makkah regions.")
                
            with c4:
                st.write("### Baseline Actual Capacity Forecast (Installed Only)")
                fig4 = plt.figure(figsize=(8, 4.5))
                # Unpack predictive results directly from your team's function
                future_2030, vision_target, trend_fit, yearly, slope = plot_forecast_by_status(df_merged, status_type='Installed', forecast_until=2030)
                st.pyplot(plt.gcf())
                plt.close(fig4)
                # 💡 User-Centric Explanation
                st.info("**AI Insight:** Our Predictive Linear Regression model utilizes historical deployment velocity to project capacity up to the year 2030, drawing a direct mathematical comparison against official national targets.")

            # 5TH GRAPH
            st.write("---")
            st.write("### 🏁 Accelerated National Pipeline Forecast (Installed + Planned Projects)")
            fig5 = plt.figure(figsize=(10, 5))
            _ = plot_forecast_by_status(df_merged, status_type='Planned', forecast_until=2030)
            st.pyplot(plt.gcf())
            plt.close(fig5)
            # 💡 User-Centric Explanation
            st.info("**AI Insight:** This model represents the future outlook by combining currently running facilities with announced upcoming projects. It shows the real statistical impact of our upcoming clean energy national pipeline towards the Vision 2030 target.")
            
            # 6️⃣ User-Centric AI Model Evaluation Report Section
            st.write("---")
            st.subheader("🎯 Automated Vision 2030 Alignment Report")
            
            # Dynamically pull metrics calculated by evaluation.py
            metrics = evaluate_forecast(future_2030, vision_target, trend_fit, yearly, slope)
            
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.success(f"**Model Trend Fit (R² Score):** {metrics['trend_fit']:.4f}")
            with ec2:
                # Fixed the function call from attention_needed to warning cleanly
                if metrics['gap'] > 0:
                    st.warning(f"**Projected Vision Gap:** {int(metrics['gap']):,} MW remaining")
                else:
                    st.success("**Target Met or Exceeded!**")
            with ec3:
                st.metric(label="Target Achievement Rate", value=f"{metrics['achievement_rate']:.2f}%")
                
            # 7️⃣ Display Dataframe at the bottom
            st.write("---")
            st.subheader("📋 Cleaned Dataset Main Preview")
            st.dataframe(df_merged, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Pipeline Execution Error: {str(e)}")
# Load and clean the raw data using team's functions (Runs once to keep app fast)
try:
    data1_raw, data2_raw = load_data()
    d1_cleaned = clean_data1(data1_raw)
    d2_cleaned = clean_data2(data2_raw)
    df_merged = merge(d1_cleaned, d2_cleaned)
    
    # 🛠️ CRITICAL COLUMNS MATCHING FIX
    if 'Capacity (MW)' in df_merged.columns:
        df_merged['Capacity'] = df_merged['Capacity (MW)']
    if 'Capacity (MW)' in d1_cleaned.columns:
        d1_cleaned['Capacity'] = d1_cleaned['Capacity (MW)']
        
except Exception as e:
    st.error(f"Backend Pipeline Initialization Error: {str(e)}")
    st.stop()

# 🛠️ SIDEBAR CONTROLS: Dynamic User Selection (Interactive Filters)
st.sidebar.header("🕹️ Interactive User Controls")
st.sidebar.write("Select filters to customize the data preview and KPIs:")

# Get unique sorted cities and years directly from your real dataset
available_cities = sorted(df_merged['City'].dropna().unique().tolist())
available_years = sorted(df_merged['Year'].dropna().unique().tolist())

# Add an "All Cities" and "All Years" option for flexibility
city_options = ["All Cities"] + available_cities
year_options = ["All Years"] + [int(y) for y in available_years]

selected_city = st.sidebar.selectbox("📍 Select City / Region:", city_options)
selected_year = st.sidebar.selectbox("📅 Select Target Year:", year_options)

# Apply User Filters dynamically to the dataframe
df_filtered = df_merged.copy()

if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered['City'] == selected_city]

if selected_year != "All Years":
    df_filtered = df_filtered[df_filtered['Year'] == selected_year]

# 3️⃣ Calculate real KPI metrics dynamically based on USER SELECTION
total_projects = len(df_filtered)
total_current_capacity = df_filtered['Capacity'].sum() if total_projects > 0 else 0
avg_capacity = df_filtered['Capacity'].mean() if total_projects > 0 else 0

# 4️⃣ Display real system KPI metrics calculated from filtered data
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Tracked Projects (Filtered)", value=f"{total_projects} Projects")
with col2:
    st.metric(label="⚡ Filtered System Capacity", value=f"{int(total_current_capacity):,} MW")
with col3:
    st.metric(label="📐 Average Project Capacity", value=f"{avg_capacity:.2f} MW" if total_projects > 0 else "0.00 MW")
    
st.write("---")

# 5️⃣ Grid Layout for the Core Team Analytics & Global AI Forecasts
st.subheader("📊 Team System Visualizations & AI Forecasts")
st.write(f"Showing global system models alongside filtered real-time statistics for **{selected_city}** in **{selected_year}**.")

# First row of charts: Growth and Solar vs Wind
c1, c2 = st.columns(2)
with c1:
    st.write("### Yearly Renewable Growth (Installed)")
    fig1 = plt.figure(figsize=(8, 4.5))
    plot_yearly_growth(d1_cleaned)
    st.pyplot(plt.gcf())
    plt.close(fig1)
    st.info("**AI Insight:** Tracks historical established renewable installations globally to preserve model dimension integrity.")
    
with c2:
    st.write("### Solar vs Wind Capacity Comparison")
    fig2 = plt.figure(figsize=(8, 4.5))
    plot_solar_vs_wind(df_merged)
    st.pyplot(plt.gcf())
    plt.close(fig2)
    st.info("**AI Insight:** Global breakdown comparing total capacity distribution between Solar and Wind projects.")
    
# Second row of charts: Regional and Vision 2030 AI Forecast
c3, c4 = st.columns(2)
with c3:
    st.write("### Regional Project Distribution")
    fig3 = plt.figure(figsize=(8, 4.5))
    plot_regional_distribution(df_merged)
    st.pyplot(plt.gcf())
    plt.close(fig3)
    st.info("**AI Insight:** Geographic summary identifying major project hosts across Saudi Arabian provinces.")
    
with c4:
    st.write("### Baseline Actual Capacity Forecast (Installed Only)")
    fig4 = plt.figure(figsize=(8, 4.5))
    # Unpack predictive results directly from your team's function
    future_2030, vision_target, trend_fit, yearly, slope = plot_forecast_by_status(df_merged, status_type='Installed', forecast_until=2030)
    st.pyplot(plt.gcf())
    plt.close(fig4)
    st.info("**AI Insight:** Our Predictive Linear Regression model projecting cumulative system capacity up to the year 2030.")

# 5TH GRAPH
st.write("---")
st.write("### 🏁 Accelerated National Pipeline Forecast (Installed + Planned Projects)")
fig5_bottom = plt.figure(figsize=(10, 5))
_ = plot_forecast_by_status(df_merged, status_type='Planned', forecast_until=2030)
st.pyplot(plt.gcf())
plt.close(fig5_bottom)
st.info("**AI Insight:** This model represents the future outlook by combining currently running facilities with announced upcoming projects. It shows the real statistical impact of our upcoming clean energy national pipeline towards the Vision 2030 target.")

# 6️⃣ User-Centric AI Model Evaluation Report Section
st.write("---")
st.subheader("🎯 Automated Vision 2030 Alignment Report")

metrics = evaluate_forecast(future_2030, vision_target, trend_fit, yearly, slope)

ec1, ec2, ec3 = st.columns(3)
with ec1:
    st.success(f"**Model Trend Fit (R² Score):** {metrics['trend_fit']:.4f}")
with ec2:
    if metrics['gap'] > 0:
        st.warning(f"**Projected Vision Gap:** {int(metrics['gap']):,} MW remaining")
    else:
        st.success("**Target Met or Exceeded!**")
with ec3:
    st.metric(label="Target Achievement Rate", value=f"{metrics['achievement_rate']:.2f}%")
    
# 7️⃣ Dynamic Interacted Dataframe Display at the bottom
st.write("---")
st.subheader(f"📋 Dataset Preview: {selected_city} ({selected_year})")
if total_projects > 0:
    st.dataframe(df_filtered, use_container_width=True)
else:
    st.warning(f"No projects found in the dataset for the selected combination: {selected_city} in {selected_year}.")

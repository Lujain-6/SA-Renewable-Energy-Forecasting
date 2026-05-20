import logging
import time
from data_loader import load_data
from preprocessing import clean_data1, clean_data2, merge
from visualization import plot_yearly_growth, plot_solar_vs_wind, plot_regional_distribution, plot_forecast_by_status
from evaluation import evaluate_dual_forecasts

def main():
    try:
        # Load Data
        data1, data2 = load_data()
        
        # Clean Data
        logging.info("Starting data cleaning phase...")
        data1_clean = clean_data1(data1)
        data2_clean = clean_data2(data2)
        
        # Merge Data
        logging.info("Merging datasets and removing duplicates...")
        df_merged = merge(data1_clean, data2_clean)
        
        # Exploratory Data Analysis (Visualizations)
        print("\n--- Generating Exploratory Analysis Charts ---")
        plot_yearly_growth(df_merged)
        plot_solar_vs_wind(df_merged)
        plot_regional_distribution(df_merged)
        
        # Model Training and Forecasting until 2030
        print("\n--- Training Models and Calculating Vision 2030 Forecasts ---")
        inst_metrics = plot_forecast_by_status(df_merged, status_type='Installed', forecast_until=2030)
        plan_metrics = plot_forecast_by_status(df_merged, status_type='Planned', forecast_until=2030)
        
        # Dual Forecast Evaluation
        evaluate_dual_forecasts(inst_metrics, plan_metrics)
        
        # Model Convergence Tests
        print("\n--- Running Model Convergence & Stability Tests ---")
        test_model_convergence(df_merged, status_type='Installed')
        test_model_convergence(df_merged, status_type='Planned')
        
        logging.info("Pipeline execution completed successfully.")
        
    except Exception as e:
        logging.critical(f"Pipeline execution failed due to an error: {e}")

if __name__ == "__main__":
    main()

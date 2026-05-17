def main():
    logging.info("Starting Renewable Energy Analysis Pipeline...")
    start_total = time.time()
    
    try:
        raw_data1, raw_data2 = load_data()
        cleaned1 = clean_data1(raw_data1)
        cleaned2 = clean_data2(raw_data2)
        df_final = merge(cleaned1, cleaned2)
        
        if df_final is None or df_final.empty:
            raise ValueError("Merged dataframe is empty. Cannot proceed.")
        
        logging.info("Generating analytical visualization dashboards...")
        plot_yearly_growth(df_final)
        plot_solar_vs_wind(df_final)
        plot_regional_distribution(df_final)
        
        # Dynamic calls: 
        # First call evaluates current actual capacity trend
        inst_metrics = plot_forecast_by_status(df_final, status_type='Installed', forecast_until=2030)
        # Second call now evaluates the global combined pipeline (Installed + Planned)
        plan_metrics = plot_forecast_by_status(df_final, status_type='Planned', forecast_until=2030)
        
        evaluate_dual_forecasts(inst_metrics, plan_metrics)
        
        total_time = time.time() - start_total
        logging.info(f"Analysis Completed Successfully in {total_time:.2f} seconds")
    except Exception as e:
        logging.critical(f"Pipeline crashed during execution workflow: {e}")
        raise e

if __name__ == "__main__":
    main()

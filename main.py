def main():

    # Load datasets
    try:
        data1_raw, data2_raw = load_data()

    except FileNotFoundError:
        print("Error: CSV files not found")
        return

    # Clean both datasets
    d1_cleaned = clean_data1(data1_raw)
    d2_cleaned = clean_data2(data2_raw)

    # Merge datasets for analysis and forecasting
    df_raw = merge(d1_cleaned, d2_cleaned)

    # Print forecast phase title
    print("Building Trend Forecast Model")

    print(
        "Forecast model uses Year → Cumulative Capacity "
        "without random train/test split."
    )

    # Print visualization phase title
    print("Generating Charts")

    # Plot yearly renewable energy growth
    plot_yearly_growth(df_raw)

    # Plot solar and wind comparison
    plot_solar_vs_wind(df_raw)

    # Plot renewable energy distribution by region
    plot_regional_distribution(df_raw)

    # Generate renewable energy forecast
    future_2030, vision_target, trend_fit, yearly, slope = plot_forecast(
        df_raw,
        forecast_until=2030
    )

    # Evaluate forecast results
    evaluate_forecast(
        future_2030=future_2030,
        vision_target=vision_target,
        trend_fit=trend_fit,
        yearly=yearly,
        slope=slope
    )

if __name__ == "__main__":
    main()

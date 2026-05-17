# Generate Yearly Capacity Growth Bar Chart
def plot_yearly_growth(df_raw):
    # Filter the dataset to include only operational (Installed) projects
    installed = df_raw[df_raw['Installed / Planned'] == 'Installed']
    
    # Group the filtered data by Year and calculate the total sum of capacity for each year
    yearly = installed.groupby('Year')['Capacity'].sum().reset_index()
    
    # Initialize the plot figure and axis with a custom size
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Generate a bar chart using the aggregated yearly data with a specific green color
    ax.bar(yearly['Year'], yearly['Capacity'], color='#2ecc71', label='Annual Added (MW)')
    
    # Configure informative text labels for both X and Y axes along with the chart title
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Capacity Added (MW)')
    ax.set_title('Saudi Arabia – Yearly Renewable Energy Growth (Installed Projects)')
    
    # Format the X-axis to display years as clean, non-decimal integer values
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Place the chart legend in the upper left corner
    ax.legend(loc='upper left')
    
    # Optimize the spacing of visual elements to prevent truncation, then save and display the plot
    plt.tight_layout()
    plt.savefig('output_yearly_growth.png', dpi=150)
    plt.show()
    
    # Log a success message confirming that the image file has been saved
    logging.info("Chart saved successfully: output_yearly_growth.png")

# Generate Solar vs Wind Energy Mix Comparison Plots with structural data validation
def plot_solar_vs_wind(df_raw):
    try:
        # Guard rail: Verify input argument is a valid pandas DataFrame
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError("Input df_raw must be a valid pandas DataFrame.")

        logging.info("Starting solar vs wind energy mix comparative analysis...")

        # Guard rail: Verify required analysis columns exist in the DataFrame before grouping
        required_cols = ['Installed / Planned', 'Type (solar/ wind)', 'Capacity']
        for col in required_cols:
            if col not in df_raw.columns:
                raise KeyError(f"Required column '{col}' is missing from the input DataFrame.")

        # Group data by status and type to compute summary capacity metrics
        comparison = df_raw.groupby(['Installed / Planned', 'Type (solar/ wind)'])['Capacity'].sum().unstack(fill_value=0)
        
        # Initialize the multi-plot figure consisting of 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left subplot: Bar tracking capacity comparisons
        comparison.plot(kind='bar', ax=axes[0], color=['#f39c12', '#3498db', '#2ecc71'])
        axes[0].set_title('Installed vs Planned Capacity by Energy Type')
        axes[0].set_xlabel('Project Status')
        axes[0].set_ylabel('Capacity (MW)')
        axes[0].tick_params(axis='x', rotation=0)
        
        # Right subplot: Share distribution pie chart showing total proportions
        totals = df_raw.groupby('Type (solar/ wind)')['Capacity'].sum()
        axes[1].pie(totals, labels=totals.index, autopct='%1.1f%%', colors=['#f39c12', '#3498db', '#2ecc71'], startangle=140)
        axes[1].set_title('Total Energy Mix Share')
        
        # Add global super title, apply tight bounding boxes, and save to directory
        plt.suptitle('Solar vs Wind Energy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output_solar_vs_wind.png', dpi=150)
        plt.show()
        
        # Log a message confirming the chart image has been successfully created
        logging.info("Chart saved successfully: output_solar_vs_wind.png")
        logging.info("Solar vs wind comparison visualization pipeline completed successfully.")

    except TypeError as e:
        logging.error(f"Data type validation error in plot_solar_vs_wind: {e}")
        raise e
    except KeyError as e:
        logging.error(f"Missing column structural error in plot_solar_vs_wind: {e}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred during solar vs wind charts generation: {e}")
        raise e

# Generate Regional Energy Distribution Chart with structural data validation
def plot_regional_distribution(df_raw):
    try:
        # Guard rail: Verify input argument is a valid pandas DataFrame
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError("Input df_raw must be a valid pandas DataFrame.")

        logging.info("Starting regional energy distribution analysis and chart generation...")

        # Guard rail: Verify required analysis columns exist in the DataFrame
        required_cols = ['City', 'Installed / Planned', 'Capacity']
        for col in required_cols:
            if col not in df_raw.columns:
                raise KeyError(f"Required column '{col}' is missing from the input DataFrame.")

        # Group data by City and project status, summing the capacity, and unstack to create separate columns
        regional = df_raw.groupby(['City', 'Installed / Planned'])['Capacity'].sum().unstack(fill_value=0)
        
        # Exclude rows belonging to 'Multi-city' projects to focus strictly on specific individual regions
        regional = regional.drop(index='Multi-city', errors='ignore')
        
        # Sort the regions in ascending order based on the 'Installed' capacity for better visualization alignment
        regional = regional.sort_values(by='Installed', ascending=True)
        
        # Initialize the plot figure and axis with a suitable size for horizontal bars
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Generate a horizontal stacked bar chart with custom green and orange colors
        regional.plot(kind='barh', stacked=True, ax=ax, color=['#2ecc71', '#f39c12'])
        
        # Set the X-axis text label and chart title
        ax.set_xlabel('Total Capacity (MW)')
        ax.set_title('Regional Renewable Energy Distribution\n(Installed vs Planned)')
        
        # Format the X-axis numbers to include thousands comma separators (e.g., 10,000)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        
        # Automatically adjust layout spacing to fit labels neatly without clipping
        plt.tight_layout()
        plt.savefig('output_regional_distribution.png', dpi=150)
        plt.show()
        
        # Log a message confirming the chart image has been successfully created
        logging.info("Chart saved successfully: output_regional_distribution.png")
        
        # Calculate the total capacity per city by summing up across both Installed and Planned categories
        total_capacity = regional.sum(axis=1).sort_values(ascending=False)
        
        # Display and print out a summary report highlighting the top 3 regions
        print("\nTop 3 regions by total capacity:")
        for i, (city, val) in enumerate(total_capacity.head(3).items(), 1):
            print(f"   {i}. {city}: {val:,.0f} MW")
            
        logging.info("Regional distribution chart and textual summary generated successfully.")

    except TypeError as e:
        logging.error(f"Data type validation error in plot_regional_distribution: {e}")
        raise e
    except KeyError as e:
        logging.error(f"Missing column structural error in plot_regional_distribution: {e}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred during regional chart generation: {e}")
        raise e

# Compute Regression Trends and Plot Forecast charts with strict data validation and error handling
def plot_forecast_by_status(df_raw, status_type, forecast_until=2030):
    try:
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError("Input df_raw must be a valid pandas DataFrame.")
        if status_type not in ['Installed', 'Planned']:
            raise ValueError("status_type must be either 'Installed' or 'Planned'.")

        logging.info(f"Starting forecasting process specifically for '{status_type}' track...")
        
        required_cols = ['Installed / Planned', 'Year', 'Capacity']
        for col in required_cols:
            if col not in df_raw.columns:
                raise KeyError(f"Required column '{col}' is missing from the input DataFrame.")
        
        # Smart dynamic filtering logic
        if status_type == 'Installed':
            filtered_df = df_raw[df_raw['Installed / Planned'] == 'Installed']
            chart_title = 'Future Renewable Energy Capacity Forecast – Installed Baseline'
            chart_color = '#2ecc71'  # Green
            line_style = '--'
        else:
            filtered_df = df_raw[df_raw['Installed / Planned'].isin(['Installed', 'Planned'])]
            chart_title = 'Future Renewable Energy Capacity Forecast – Combined (Installed + Planned)'
            chart_color = '#f39c12'  # Orange/Amber
            line_style = '-.'
            
        # Sort by year first, group, then apply .cumsum() for cumulative capacity tracking
        yearly = filtered_df.groupby('Year')['Capacity'].sum().sort_index().cumsum().reset_index()
        
        if yearly.empty:
            raise ValueError(f"No project data found for execution track: '{status_type}'.")
            
        X = yearly['Year'].values.reshape(-1, 1)
        y = yearly['Capacity'].values
        
        model = train_renewable_model(X, y)
        future_years = np.arange(yearly['Year'].min(), forecast_until + 1).reshape(-1, 1)
        predictions = model.predict(future_years)
        
        vision_target = 58700
        future_2030 = model.predict([[2030]])[0]
        
        # Plotting configuration
        fig, ax = plt.subplots(figsize=(11, 6))
        if status_type == 'Installed':
            data_label = 'Cumulative Data (Installed)'
        else:
            data_label = 'Cumulative Data (Installed + Planned)'
        # Plot continuous cumulative line + scatter points
        ax.plot(yearly['Year'], yearly['Capacity'], color=chart_color, linewidth=2, marker='o', label=data_label)
        ax.plot(future_years, predictions, color='#2c3e50', linewidth=2, linestyle=line_style, label= 'Model Prediction Line')
        ax.axhline(y=vision_target, color='#e74c3c', linewidth=1.8, linestyle=':', label=f'Vision 2030 Target ({vision_target:,} MW)')
        
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Total Cumulative Capacity (MW)', fontweight='bold')
        ax.set_title(chart_title, fontweight='bold', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        output_filename = f'output_forecast_{status_type.lower()}.png'
        plt.savefig(output_filename, dpi=150)
        plt.show()
        
        logging.info(f"Forecasting completed and cumulative asset saved: {output_filename}")
        return future_2030, vision_target, model.score(X, y), model.coef_[0]
    except Exception as e:
        logging.error(f"Error occurred during {status_type} forecasting: {e}")
        raise e


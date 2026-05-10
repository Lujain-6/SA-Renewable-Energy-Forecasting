# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
# YEARLY GROWTH CHART
def plot_yearly_growth(df_raw):
    """
    df_raw: the ORIGINAL (unscaled) merged dataset before StandardScaler.
    Expects columns: 'Year', 'Capacity', 'Installed / Planned'
    """
    # Filter the dataset to include only projects that are already 'Installed'
    installed = df_raw[df_raw['Installed / Planned'] == 'Installed']
    
    # Group data by year and sum the capacity to get total added MW per year
    yearly = installed.groupby('Year')['Capacity'].sum().reset_index()
    
    # Calculate the cumulative sum of capacity across the years
    yearly['Cumulative_MW'] = yearly['Capacity'].cumsum()
    
    # Create a figure and axis for the plot with a specific size
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot annual added capacity as a bar chart (primary Y-axis)
    ax.bar(yearly['Year'], yearly['Capacity'], color='#2ecc71', label='Annual Added (MW)')

    # Create a twin Y-axis to share the same X-axis (for the cumulative line)
    ax2 = ax.twinx()

    # Plot cumulative capacity as a line chart with markers (secondary Y-axis)
    ax2.plot(yearly['Year'], yearly['Cumulative_MW'], color='#2c3e50',
             marker='o', linewidth=2, label='Cumulative (MW)')
    
    # Set labels for the X-axis and both Y-axes
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Capacity Added (MW)')
    ax2.set_ylabel('Cumulative Capacity (MW)')
    ax.set_title('Saudi Arabia – Yearly Renewable Energy Growth (Installed Projects)')
    
    # Ensure the X-axis only displays integer values for years (no decimals)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Format the X-axis labels to display as plain year integers
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Collect legend handles and labels from both axes to merge them into one legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    plt.savefig('output_yearly_growth.png', dpi=150)
    plt.show()
    # SOLAR VS WIND COMPARISON  
def plot_solar_vs_wind(df_raw):
    """
    Expects columns: 'Type (solar/ wind)', 'Capacity', 'Year', 'Installed / Planned'
    """
    # Group data by status and energy type, then sum capacities and pivot for plotting
    comparison = df_raw.groupby(
        ['Installed / Planned', 'Type (solar/ wind)']
    )['Capacity'].sum().unstack(fill_value=0)

    # Initialize a figure with two side-by-side subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot a grouped bar chart on the first subplot (Installed vs Planned)
    comparison.plot(
        kind='bar',
        ax=axes[0],
        color=['#f39c12', '#3498db', '#2ecc71']
    )

    # Set aesthetics for the first subplot
    axes[0].set_title('Installed vs Planned Capacity by Energy Type')
    axes[0].set_xlabel('Project Status')
    axes[0].set_ylabel('Capacity (MW)')
    axes[0].tick_params(axis='x', rotation=0)

    # Group data by energy type only to calculate the overall mix for the pie chart
    totals = df_raw.groupby(
        'Type (solar/ wind)'
    )['Capacity'].sum()

    # Create a pie chart on the second subplot to show percentage distribution
    axes[1].pie(
        totals,
        labels=totals.index,
        autopct='%1.1f%%',
        colors=['#f39c12', '#3498db', '#2ecc71'],
        startangle=140)

    # Set the title for the pie chart
    axes[1].set_title('Total Energy Mix Share')

    # Add a main centralized title for the entire visualization
    plt.suptitle(
        'Solar vs Wind Energy Comparison',
        fontsize=14,
        fontweight='bold')

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()

    plt.savefig(
        'output_solar_vs_wind.png',
        dpi=150)

    plt.show()

    print("Chart saved: output_solar_vs_wind.png")
    ## REGIONAL DISTRIBUTION
def plot_regional_distribution(df_raw):

    """
    Expects columns:
    'City',
    'Capacity',
    'Installed / Planned'
    """

    # Group renewable capacity by city and project status
    regional = df_raw.groupby(
        ['City', 'Installed / Planned']
    )['Capacity'].sum().unstack(fill_value=0)

    # Remove multi-city projects
    regional = regional.drop(
        index='Multi-city',
        errors='ignore'
    )

    # Sort regions by installed capacity
    regional = regional.sort_values(
        by='Installed',
        ascending=True
    )

    # Create chart figure
    fig, ax = plt.subplots(figsize=(11, 7))

    # Create stacked horizontal bar chart
    regional.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=['#2ecc71', '#f39c12']
    )

    # Set x-axis label
    ax.set_xlabel('Total Capacity (MW)')

    # Set chart title
    ax.set_title(
        'Regional Renewable Energy Distribution\n'
        '(Installed vs Planned)'
    )

    # Format x-axis values
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )

    # Adjust layout
    plt.tight_layout()

    # Save chart image
    plt.savefig(
        'output_regional_distribution.png',
        dpi=150
    )

    # Display chart
    plt.show()

    # Print saved chart message
    print("Chart saved: output_regional_distribution.png")

    # Calculate total renewable capacity for each city
    total_capacity = regional.sum(
        axis=1
    ).sort_values(ascending=False)

    # Print top regions title
    print("\nTop 3 regions by total capacity:")

    # Print top 3 regions
    for i, (city, val) in enumerate(
        total_capacity.head(3).items(),
        1
    ):

        print(
            f"   {i}. {city}: {val:,.0f} MW"
        )

    # Plot renewable energy forecast until 2030
def plot_forecast(df_raw, forecast_until=2030):

    # Keep only installed renewable projects
    installed = df_raw[
        df_raw['Installed / Planned'] == 'Installed'
    ]

    # Calculate yearly renewable capacity
    yearly = installed.groupby(
        'Year'
    )['Capacity'].sum().reset_index()

    # Calculate cumulative renewable capacity
    yearly['Cumulative_MW'] = yearly[
        'Capacity'
    ].cumsum()

    # Prepare years for model training
    X = yearly['Year'].values.reshape(-1, 1)

    # Prepare cumulative capacity target
    y = yearly['Cumulative_MW'].values

    # Train renewable forecasting model
    model = train_renewable_model(X, y)

    # Create future years until 2030
    future_years = np.arange(
        yearly['Year'].min(),
        forecast_until + 1
    ).reshape(-1, 1)

    # Generate future predictions
    predictions = model.predict(
        future_years
    )

    # Vision 2030 renewable energy target (~58.7 GW)
    vision_target = 58700

    # Predict renewable capacity in 2030
    future_2030 = model.predict(
        [[2030]]
    )[0]

    # Calculate remaining gap
    gap = vision_target - future_2030

    # Create chart figure
    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot historical renewable capacity
    ax.scatter(
        yearly['Year'],
        yearly['Cumulative_MW'],
        color='#2ecc71',
        zorder=5,
        s=60,
        label='Historical (Installed)'
    )

    # Plot forecast trend line
    ax.plot(
        future_years,
        predictions,
        color='#2c3e50',
        linewidth=2,
        linestyle='--',
        label='Linear Regression Forecast'
    )

    # Plot Vision 2030 target line
    ax.axhline(
        y=vision_target,
        color='#e74c3c',
        linewidth=1.8,
        linestyle=':',
        label=f'Vision 2030 Target ({vision_target:,} MW)'
    )

    # Add gap annotation
    ax.annotate(
        f'Gap in 2030:\n{gap:,.0f} MW',
        xy=(2030, future_2030),
        xytext=(2027, future_2030 + 5000),
        arrowprops=dict(
            arrowstyle='->',
            color='black'
        ),
        fontsize=9,
        color='#c0392b'
    )

    # Set x-axis label
    ax.set_xlabel('Year')

    # Set y-axis label
    ax.set_ylabel('Cumulative Capacity (MW)')

    # Set chart title
    ax.set_title(
        'Future Renewable Energy Capacity Forecast – Saudi Arabia'
    )

    # Display chart legend
    ax.legend()

    # Format y-axis values
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )

    # Adjust layout
    plt.tight_layout()

    # Save chart image
    plt.savefig('output_forecast.png', dpi=150)

    # Display chart
    plt.show()

    # Return forecast values for evaluation
    return (
        future_2030,
        vision_target,
        model.score(X, y),
        yearly,
        model.coef_[0]
    )
    print("Chart saved: output_yearly_growth.png")


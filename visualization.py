# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression

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

    # Create Linear Regression model
    model = LinearRegression()

    # Train forecasting model
    model.fit(X, y)

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

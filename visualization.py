# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression

# 1. YEARLY GROWTH CHART
def plot_yearly_growth(df_raw):
    """
    df_raw: the ORIGINAL (unscaled) merged dataset before StandardScaler.
    Expects columns: 'Year', 'Capacity', 'Installed / Planned'
    """
    installed = df_raw[df_raw['Installed / Planned'] == 'Installed']
    yearly = installed.groupby('Year')['Capacity'].sum().reset_index()
    yearly['Cumulative_MW'] = yearly['Capacity'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly['Year'], yearly['Capacity'], color='#2ecc71', label='Annual Added (MW)')
    ax2 = ax.twinx()
    ax2.plot(yearly['Year'], yearly['Cumulative_MW'], color='#2c3e50',
             marker='o', linewidth=2, label='Cumulative (MW)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Capacity Added (MW)')
    ax2.set_ylabel('Cumulative Capacity (MW)')
    ax.set_title('Saudi Arabia – Yearly Renewable Energy Growth (Installed Projects)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig('output_yearly_growth.png', dpi=150)
    plt.show()
    print("Chart saved: output_yearly_growth.png")


# 2. SOLAR VS WIND COMPARISON
def plot_solar_vs_wind(df_raw):
    """
    Expects columns: 'Type (solar/ wind)', 'Capacity', 'Year', 'Installed / Planned'
    """
    installed = df_raw[df_raw['Installed / Planned'] == 'Installed']
    comparison = installed.groupby(['Year', 'Type (solar/ wind)'])['Capacity'].sum().unstack(fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar
    comparison.plot(kind='bar', stacked=True, ax=axes[0],
                    color=['#f39c12', '#3498db', '#2ecc71'])
    axes[0].set_title('Solar vs Wind – Annual Capacity by Year')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Capacity (MW)')
    axes[0].tick_params(axis='x', rotation=45)

    # Pie chart (total share)
    totals = installed.groupby('Type (solar/ wind)')['Capacity'].sum()
    axes[1].pie(totals, labels=totals.index, autopct='%1.1f%%',
                colors=['#f39c12', '#3498db', '#2ecc71'], startangle=140)
    axes[1].set_title('Total Energy Mix Share')

    plt.suptitle('Solar vs Wind Energy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_solar_vs_wind.png', dpi=150)
    plt.show()
    print("Chart saved: output_solar_vs_wind.png")


# 3. REGIONAL DISTRIBUTION
def plot_regional_distribution(df_raw):
    """
    Expects columns: 'City', 'Capacity', 'Installed / Planned'
    """
    regional = df_raw.groupby('City')['Capacity'].sum().sort_values(ascending=True)

    # Highlight top 3 fastest-growing regions
    colors = ['#e74c3c' if city in regional.nlargest(3).index else '#3498db'
              for city in regional.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(regional.index, regional.values, color=colors)
    ax.set_xlabel('Total Capacity (MW)')
    ax.set_title('Regional Distribution of Renewable Energy Capacity\n'
                 '(Red = Top 3 fastest-growing regions)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    plt.savefig('output_regional_distribution.png', dpi=150)
    plt.show()
    print("Chart saved: output_regional_distribution.png")


# 4. FUTURE CAPACITY FORECAST (Vision 2030)
def plot_forecast(df_raw, forecast_until=2030):
    """
    Trains a simple Linear Regression on yearly cumulative capacity
    and forecasts until forecast_until year.
    Shows Vision 2030 target line.
    """
    installed = df_raw[df_raw['Installed / Planned'] == 'Installed']
    yearly = installed.groupby('Year')['Capacity'].sum().reset_index()
    yearly['Cumulative_MW'] = yearly['Capacity'].cumsum()

    X = yearly['Year'].values.reshape(-1, 1)
    y = yearly['Cumulative_MW'].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(yearly['Year'].min(), forecast_until + 1).reshape(-1, 1)
    predictions = model.predict(future_years)

    # Vision 2030 target: 50% of ~120 GW total capacity ≈ 60,000 MW
    VISION_TARGET_MW = 60000

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(yearly['Year'], yearly['Cumulative_MW'],
               color='#2ecc71', zorder=5, s=60, label='Historical (Installed)')
    ax.plot(future_years, predictions,
            color='#2c3e50', linewidth=2, linestyle='--', label='Linear Regression Forecast')
    ax.axhline(y=VISION_TARGET_MW, color='#e74c3c', linewidth=1.8,
               linestyle=':', label=f'Vision 2030 Target ({VISION_TARGET_MW:,} MW)')

    # Annotate gap in 2030
    pred_2030 = model.predict([[2030]])[0]
    gap = VISION_TARGET_MW - pred_2030
    ax.annotate(f'Gap in 2030:\n{gap:,.0f} MW',
                xy=(2030, pred_2030), xytext=(2027, pred_2030 + 5000),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, color='#c0392b')

    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Capacity (MW)')
    ax.set_title('Future Renewable Energy Capacity Forecast – Saudi Arabia\n'
                 f'R² = {model.score(X, y):.2f}  |  Slope = {model.coef_[0]:,.0f} MW/year')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    plt.savefig('output_forecast.png', dpi=150)
    plt.show()
    print(f"Chart saved: output_forecast.png")
    print(f"Predicted capacity in 2030: {pred_2030:,.0f} MW")
    print(f"Gap from Vision 2030 target: {gap:,.0f} MW")

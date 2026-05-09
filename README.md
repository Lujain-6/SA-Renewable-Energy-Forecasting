# Renewable Energy Forecasting - SA

**Predicting Renewable Energy Capacity in Saudi Arabia using Machine Learning.**

This project provides a software framework to analyze renewable energy data (Solar, Wind, etc.) in the Kingdom of Saudi Arabia, It features a predictive model to forecast cumulative production capacities and evaluate progress toward national energy targets.

## Overview

The project focuses on processing two main types of data:

* Installed Projects: Current real-world operational capacities.

* Planned Projects: Targeted future expansions and upcoming installations.

The model utilizes the Linear Regression algorithm to establish relationships between various factors and the resulting energy capacity.

## Code Structure
| Function | Descreption |
| --- | --- |
| load_data() | load the datasets from a local directory (KAPSARC and Data Saudi) |
| clean_data1(data1) | Handle the Missing Values and drop the unnecessary Columns |
| clean_data2(data2) | drop the unnecessary Columns, add city column and rename some columns to match data1 |
| merge_and_encode(data1, data2) | stack the cleaned datasets. then Standardize and encode the columns to ensure optimal model training |
| train_renewable_model(X, y) | use Linear Regression to train the model on historical project data to forecast the expected energy capacity |
| evaluate_forecast(future_2030, vision_target, trend_fit, yearly, slope) | Outputs the final model metrics (R^2 score) and calculates the achievement gap toward Vision 2030 |
## Getting Started

 [1] Prerequisites
 
  Ensure you have the following packages installed:
  
  * pip install pandas scikit-learn
  
 [2] Data Setup
 
  Place the following files in a folder named data/:
  
  * saudi-arabia-planned-and-installed-renewables-by-project.csv
  * renewable_energy_projects.csv

## Tech Stack

* Pandas: For large-scale data manipulation and energy dataset cleaning.
* Scikit-learn: For building and evaluating predictive machine learning models.
* Linear Regression: The primary algorithm used for capacity forecasting.
* Matplotlib: For generating regional distribution charts and forecast visualizations.

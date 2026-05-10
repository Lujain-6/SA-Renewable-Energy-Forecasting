import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data1(data1):
    # Fill missing city values manually
    data1.loc[[2, 18], 'City'] = 'Jeddah'
    data1.loc[29, 'City'] = 'Rabigh'
    data1.loc[56, 'City'] = 'Qurayyat'
    data1.loc[[24, 66], 'City'] = 'Multi-city'
    mapping = {
    'Wadi ad-Dawasir': 'Wadi Aldawaser',
    'Sakaka': 'Sakakah',
    'Dumat al-Jandal': 'Domat al-Jandal',
    'Alhenakiyah': 'Al Henakiyah',
    'AL Hinakiyah': 'Al Henakiyah',
    'Tubarjal':'Tabarjal'
    }

    data1['City'] = data1['City'].replace(mapping)

    # Drop rows with remaining missing values
    data1 = data1.drop([19, 57])
    # Drop unnecessary columns
    data1 = data1.drop(columns=['Location', 'Source'])

    return data1


def clean_data2(data2):
    # Drop unnecessary columns
    data2 = data2.drop(columns=['Energy Type ID','Renewable Energy Project ID',
                                '2024 ID',
                                'Estimated number of housing units to be supplied with electricity',
                                'Levelized Cost of Energy (LCOE)', 'Total Investment'])
    # Rename columns to match data1
    data2 = data2.rename(columns={
        '2024': 'Year',
        'Renewable Energy Project': 'Project name',
        'Project Capacity': 'Capacity',
        'Energy Type': 'Type (solar/ wind)'
    })
    # Add City column
    data2['City'] = data2['Project name'].str.replace(r' 1$', '', regex=True)
    data2.loc[[5, 8], 'City'] = 'Jeddah'
    # Assign Installed/Planned based on year
    data2['Installed / Planned'] = np.where(
        data2['Operation Start Year'] < 2026, 'Installed', 'Planned'
    )
    data2 = data2.drop(columns=['Operation Start Year'])
    return data2

def merge(data1, data2):
    # Merge both datasets
    Dataset = pd.concat([data2, data1], axis=0).reset_index(drop=True)
    # Clean energy type values
    Dataset['Type (solar/ wind)'] = Dataset['Type (solar/ wind)'].str.replace(' energy', '')
    Dataset.columns = Dataset.columns.str.strip()
    # Remove duplicates and sort by year
    Dataset = Dataset.drop_duplicates(subset=['Type (solar/ wind)', 'Year', 'Capacity', 'City'])
    Dataset = Dataset.sort_values(by='Year', ascending=False).reset_index(drop=True)
    # Drop project name column
    Dataset = Dataset.drop(['Project name'], axis=1)

    return Dataset

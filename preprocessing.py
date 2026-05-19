import logging
import numpy as np
import pandas as pd

# Data Cleaning and Standardization for Dataset 1
def clean_data1(data1):
    try:
        # Check if the required 'City' column exists
        if 'City' not in data1.columns:
            raise KeyError("The column 'City' is missing from Dataset 1.")
            
        # Correct city misalignments safely by verifying index existence first
        target_indices = [2, 18, 29, 56, 24, 66]
        for idx in target_indices:
            if idx not in data1.index:
                logging.warning(f"Index {idx} not found in Dataset 1. Skipping alignment.")
        # Check specific row indices and manually update or correct the city names if they exist        
        if 2 in data1.index: data1.loc[2, 'City'] = 'Jeddah'
        if 18 in data1.index: data1.loc[18, 'City'] = 'Jeddah'
        if 29 in data1.index: data1.loc[29, 'City'] = 'Rabigh'
        if 56 in data1.index: data1.loc[56, 'City'] = 'Qurayyat'
        if 24 in data1.index: data1.loc[24, 'City'] = 'Multi-city'
        if 66 in data1.index: data1.loc[66, 'City'] = 'Multi-city'
        
        # Standardize city names naming syntax
        mapping = {
            'Wadi ad-Dawasir': 'Wadi Aldawaser',
            'Sakaka': 'Sakakah',
            'Dumat al-Jandal': 'Domat al-Jandal',
            'Alhenakiyah': 'Al Henakiyah',
            'AL Hinakiyah': 'Al Henakiyah',
            'Tubarjal': 'Tabarjal'
        }
        data1['City'] = data1['City'].replace(mapping)
        
        # Drop rows containing invalid structural entries safely
        rows_to_drop = [idx for idx in [19, 57] if idx in data1.index]
        if rows_to_drop:
            data1 = data1.drop(rows_to_drop)
            
        # Drop unneeded columns safely by checking if they exist
        cols_to_drop = [col for col in ['Location', 'Source'] if col in data1.columns]
        if cols_to_drop:
            data1 = data1.drop(columns=cols_to_drop)
            
        return data1
        
    except KeyError as e:
        logging.error(f"Column Structure Error in clean_data1: {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error during clean_data1 execution: {e}")
        raise e


# Data Cleaning and Standardization for Dataset 2
def clean_data2(data2):
    try:
        # Drop redundant tracking IDs and columns safely
        cols_to_drop = [
            'Energy Type ID', 'Renewable Energy Project ID', '2024 ID',
            'Estimated number of housing units to be supplied with electricity',
            'Levelized Cost of Energy (LCOE)', 'Total Investment'
        ]
        # Identify which target columns actually exist in data2 to prevent drop errors
        existing_cols = [col for col in cols_to_drop if col in data2.columns]
        data2 = data2.drop(columns=existing_cols)
        
        # Rename columns to match Dataset 1 nomenclature
        rename_mapping = {
            '2024': 'Year',
            'Renewable Energy Project': 'Project name',
            'Project Capacity': 'Capacity',
            'Energy Type': 'Type (solar/ wind)'
        }
        data2 = data2.rename(columns=rename_mapping)
        
        # Verify required columns exist after rename
        required_cols = ['Project name', 'Operation Start Year']
        for col in required_cols:
            if col not in data2.columns:
                raise KeyError(f"Required column '{col}' is missing from Dataset 2.")
                
        # Clean city naming strings derived from project titles
        data2['City'] = data2['Project name'].str.replace(r' 1$', '', regex=True)
        
        if 5 in data2.index: data2.loc[5, 'City'] = 'Jeddah'
        if 8 in data2.index: data2.loc[8, 'City'] = 'Jeddah'
        
        # Set conditional threshold for project status based on operational timeline
        data2['Installed / Planned'] = np.where(data2['Operation Start Year'] < 2026, 'Installed', 'Planned')
        data2 = data2.drop(columns=['Operation Start Year'])
        
        return data2
        
    except KeyError as e:
        logging.error(f"Column Structure Error in clean_data2: {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error during clean_data2 execution: {e}")
        raise e

# Merge Dataframes and remove residual duplicates
def merge(data1, data2):
    # Concatenate dataframes along the index row axis
    Dataset = pd.concat([data2, data1], axis=0).reset_index(drop=True)
    Dataset.columns = Dataset.columns.str.strip()
    
    # Normalize clean string values for categorical evaluation
    Dataset['Type (solar/ wind)'] = Dataset['Type (solar/ wind)'].astype(str).str.replace(' energy', '', case=False).str.strip().str.title()
    Dataset['Installed / Planned'] = Dataset['Installed / Planned'].astype(str).str.strip().str.title()
    
    # Deduplicate matching entry metrics and sort chronologically
    Dataset = Dataset.drop_duplicates(subset=['Type (solar/ wind)', 'Year', 'Capacity', 'City'])
    Dataset = Dataset.sort_values(by='Year', ascending=False).reset_index(drop=True)
    Dataset = Dataset.dropna()
    
    if 'Project name' in Dataset.columns:
        Dataset = Dataset.drop(['Project name'], axis=1)
    return Dataset

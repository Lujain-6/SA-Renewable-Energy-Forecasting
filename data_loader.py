# This library provides simple and efficient tools for predictive data analysis
#!pip install scikit-learn

# pytest: A framework to write and run automated unit tests for our code functions.
#!pip install pytest

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression

# Configure logging to output to both a local log file and the console stream
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_analysis.log'),
        logging.StreamHandler()
    ]
)

# Load datasets safely with robust file and parse checks
def load_data():
    logging.info("Dataset Loading...")
    try:
        file1_path = "data/saudi-arabia-planned-and-installed-renewables-by-project.csv"
        file2_path = "data/renewable_energy_projects.csv"
        
        # Verify if both dataset paths exist before attempting to read
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            raise FileNotFoundError("One or both dataset CSV files are missing in the 'data/' folder.")
            
        data1 = pd.read_csv(file1_path, sep=';')
        data2 = pd.read_csv(file2_path, sep=',')
        
        logging.info(f"Dataset 1 loaded successfully with {len(data1)} rows.")
        logging.info(f"Dataset 2 loaded successfully with {len(data2)} rows.")
        return data1, data2
        
    except FileNotFoundError as e:
        logging.error(f"Critical File Error: {e}")
        raise e
    except pd.errors.ParserError as e:
        logging.error(f"CSV Parsing Error (Check structural separators or delimiters): {e}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error while loading data: {e}")
        raise e

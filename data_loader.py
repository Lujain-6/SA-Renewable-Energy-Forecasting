# This library provides simple and efficient tools for predictive data analysis
!pip install scikit-learn

# pytest: A framework to write and run automated unit tests for our code functions.
!pip install pytest

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

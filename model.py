import logging
import time
from sklearn.linear_model import LinearRegression

# Train the analytical forecasting regression model
def train_renewable_model(X, y):
    logging.info("Training Linear Regression model...")
    start = time.time()
    model = LinearRegression()
    model.fit(X, y)
    logging.info(f"Model trained successfully in {time.time() - start:.2f} seconds")
    return model

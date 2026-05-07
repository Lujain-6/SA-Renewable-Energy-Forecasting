from sklearn.linear_model import LinearRegression

#This function trains the model using the training data
def train_renewable_model(X_train, y_train):
    # Training a Linear Regression model for energy capacity forecasting
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

#This function generates predictions for the evaluation stage
def get_predictions(model, X_test):
    return model.predict(X_test)
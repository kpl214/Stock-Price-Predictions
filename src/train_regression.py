from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Compute MSE
    mse = mean_squared_error(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "mse": mse
    }

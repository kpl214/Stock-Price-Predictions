from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Regression Model MSE: {mse}")

    return {
        "model": model,
        "predictions": predictions,
        "mse": mse
    }

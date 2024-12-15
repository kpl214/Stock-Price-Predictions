from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1):
    # Initialize the model
    # n_estimators is number of boosting rounds
    # max_depth is the max depth of the decision trees
    # learning_rate is step size shrinkage
    # n_jobs is the number of CPU threads to use
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=n_jobs
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Compute MSE
    mse = mean_squared_error(y_test, predictions)
    print(f"XGBoost Model MSE: {mse}")

    return {
        "model": model,
        "predictions": predictions,
        "mse": mse
    }

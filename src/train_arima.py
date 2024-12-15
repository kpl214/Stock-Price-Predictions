from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

def train_arima(weekly_data, forecast_steps=1, seasonal=False, m=1, p_range=(0, 3), q_range=(0, 3), d_range=(0, 2)):

    if len(weekly_data) == 0:
        raise ValueError("weekly_data is empty. Cannot train ARIMA model.")

    # Use auto_arima with explicit ranges for p, d, q
    auto_model = auto_arima(
        weekly_data,
        start_p=p_range[0], max_p=p_range[1],
        start_q=q_range[0], max_q=q_range[1],
        start_d=d_range[0], max_d=d_range[1],
        seasonal=seasonal,
        m=m,
        trace=True,           # Set to True to see the parameter search process
        error_action='ignore', 
        suppress_warnings=True,
        stepwise=True
    )

    # Extract the order parameters
    # p = Order of Auto-Regression - observes past values to predict current values
    # d = Degree of Differencing - determines the number of time the data is differenced consecutively
    # q = Order of Moving Average - determines past forecast errors
    # AKA very computationally expensive
    
    p, d, q = auto_model.order 
    if seasonal:
        P, D, Q, m = auto_model.seasonal_order
    else:
        P, D, Q, m = (0, 0, 0, 1)  # dummy if not seasonal

    # Fit the final ARIMA model using statsmodels with the chosen order
    model = ARIMA(weekly_data, order=(p, d, q), seasonal_order=(P, D, Q, m) if seasonal else None)
    model_fit = model.fit()

    # In-sample predictions (one-step ahead)
    start = 0
    end = len(weekly_data) - 1
    train_predictions = model_fit.predict(start=start, end=end)

    # Compute MSE on training set
    mse = mean_squared_error(weekly_data.values, train_predictions.values)

    # Forecast future values
    forecast = model_fit.forecast(steps=forecast_steps)

    return {
        "model": model_fit,
        "order": (p, d, q),
        "seasonal_order": (P, D, Q, m) if seasonal else None,
        "train_predictions": train_predictions,
        "forecast": forecast,
        "mse": mse
    }

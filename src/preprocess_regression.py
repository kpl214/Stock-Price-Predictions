import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_for_regression(file_name):
    data = pd.read_csv(file_name)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    data.set_index("timestamp", inplace=True)

    # Compute daily features
    data["SMA_20"] = data["close"].rolling(window=10).mean()
    data["SMA_50"] = data["close"].rolling(window=20).mean()
    data["daily_return"] = data["close"].pct_change()
    data["volatility"] = data["close"].rolling(window=10).std()
    data["lag_close"] = data["close"].shift(1)

    # Drop rows with NaNs from rolling calculations
    data = data.dropna()

    # Get weekly (end-of-week) close: for instance, use Friday as the week end
    weekly_close = data["close"].resample("W-FRI").last()

    # Shift weekly_close by -1 so for each current week's data, we know next week's close
    # This assumes we want to predict the upcoming week's closing price.
    weekly_close = weekly_close.shift(-1)

    # Join this weekly target back to the daily data
    data = data.join(weekly_close.rename("weekly_target"), how="left")

    # Drop rows where weekly_target is NaN (end of dataset)
    data = data.dropna()

    # Our features remain daily but the target is now the next week's closing price
    X = data[["SMA_20", "SMA_50", "daily_return", "volatility", "lag_close"]]
    y = data["weekly_target"]

    # Scale features (not target)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_for_lstm(file_name, sequence_length=5):
    data = pd.read_csv(file_name)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    data.set_index("timestamp", inplace=True)

    # Resample data to weekly frequency (e.g., last known close of the week)
    weekly_data = data.resample("W-FRI").last().dropna()

    # If desired, compute weekly-level features (example: just close for now)
    # Additional weekly features could be rolling weekly returns, weekly volatility, etc.
    # For simplicity, let's just use 'close' as a single feature here.
    close_prices = weekly_data["close"].values.reshape(-1, 1)

    # Scale the weekly close prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)

    sequences = []
    targets = []
    # Create sequences of weekly data
    for i in range(sequence_length, len(scaled_close)):
        seq = scaled_close[i-sequence_length:i]   # Past N weeks
        target = scaled_close[i]                  # Next week's close
        sequences.append(seq)
        targets.append(target)

    X_seq = np.array(sequences) # shape: (samples, sequence_length, 1)
    y_seq = np.array(targets)   # shape: (samples, 1)

    return X_seq, y_seq, scaler

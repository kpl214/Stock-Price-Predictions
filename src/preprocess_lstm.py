import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def preprocess_for_lstm(file_name, sequence_length=5):
    data = pd.read_csv(file_name)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")

    # Resample to weekly frequency
    weekly_data = data.resample("W-FRI").last().dropna()

    # Compute additional features
    weekly_data['SMA_5'] = weekly_data['close'].rolling(window=5).mean()
    weekly_data['EMA_12'] = weekly_data['close'].ewm(span=12).mean()
    weekly_data['EMA_26'] = weekly_data['close'].ewm(span=26).mean()
    weekly_data['MACD'] = weekly_data['EMA_12'] - weekly_data['EMA_26']
    weekly_data['Signal_Line'] = weekly_data['MACD'].ewm(span=9).mean()
    weekly_data['RSI'] = compute_rsi(weekly_data['close'], window=14)
    weekly_data = weekly_data.dropna()

    # Combine columns into one DataFrame or array
    # (close is col 0, then SMA_5, MACD, Signal_Line, RSI)
    close_col = weekly_data[['close']]
    other_cols = weekly_data[['SMA_5','MACD','Signal_Line','RSI']]
    # Fit ONE scaler across all 5 columns
    close_scaler = MinMaxScaler()
    scaled_close = close_scaler.fit_transform(close_col)

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(other_cols)

    combined_scaled = np.hstack([scaled_close, scaled_features])
    # Create sequences of length `sequence_length`
    X_seqs, y_seqs = [], []

    for i in range(sequence_length, len(combined_scaled)):
        # Past 'sequence_length' rows (shape: (sequence_length, 5))
        X_seqs.append(combined_scaled[i-sequence_length:i])
        # Next week's close (column 0)
        y_seqs.append(combined_scaled[i, 0])

    X_seq = np.array(X_seqs)               # shape => (samples, seq_len, 5)
    y_seq = np.array(y_seqs).reshape(-1,1) # shape => (samples, 1)

    return X_seq, y_seq, close_scaler, feature_scaler

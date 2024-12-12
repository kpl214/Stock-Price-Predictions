from api_fetch import fetch_stock_data
from preprocess_regression import preprocess_for_regression
from preprocess_lstm import preprocess_for_lstm
from train_regression import train_regression
from train_lstm import train_lstm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

def run_prediction(symbol):
    try:
        # Fetch Data
        file_name = fetch_stock_data(symbol, outputsize="full")

        # Preprocess for regression (daily data, weekly target)
        X_reg, y_reg, scaler_reg = preprocess_for_regression(file_name)

        # Preprocess for LSTM (weekly data, weekly target)
        X_lstm, y_lstm, scaler_lstm = preprocess_for_lstm(file_name, sequence_length=5)

        print("Shapes after preprocessing:")
        print("X_reg:", X_reg.shape, "y_reg:", y_reg.shape)
        print("X_lstm:", X_lstm.shape, "y_lstm:", y_lstm.shape)

        # Debugging data insufficiency
        if len(X_reg) < 20:
            print("Warning: Insufficient data for regression model.")
        if len(X_lstm) < 20:
            print("Warning: Insufficient data for LSTM model.")

        # Split regression data
        train_size_reg = int(0.8 * len(X_reg))
        X_train_reg, X_test_reg = X_reg[:train_size_reg], X_reg[train_size_reg:]
        y_train_reg, y_test_reg = y_reg[:train_size_reg], y_reg[train_size_reg:]

        # Split LSTM data
        train_size_lstm = int(0.8 * len(X_lstm))
        X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
        y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

        # Create validation set for LSTM
        val_size_lstm = int(0.1 * train_size_lstm)
        X_val_lstm = X_train_lstm[-val_size_lstm:]
        y_val_lstm = y_train_lstm[-val_size_lstm:]
        X_train_lstm = X_train_lstm[:-val_size_lstm]
        y_train_lstm = y_train_lstm[:-val_size_lstm]

        # Train LSTM
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        lstm_model = train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, input_shape, epochs=100)

        # LSTM predictions
        lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
        lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions_scaled)
        y_test_lstm_original = scaler_lstm.inverse_transform(y_test_lstm)

        lstm_mse = mean_squared_error(y_test_lstm_original, lstm_predictions)
        print(f"LSTM Model MSE (original scale): {lstm_mse}")

        # Train Regression
        regression_results = train_regression(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
        regression_predictions = regression_results["predictions"]
        regression_mse = mean_squared_error(y_test_reg, regression_predictions)

        # Combine predictions
        min_test_size = min(len(y_test_reg), len(y_test_lstm_original))
        y_test_reg_adj = y_test_reg[:min_test_size]
        regression_predictions_adj = regression_predictions[:min_test_size]
        lstm_predictions_adj = lstm_predictions[:min_test_size].flatten()

        alpha, beta = 0.7, 0.3
        combined_predictions = (alpha * regression_predictions_adj + beta * lstm_predictions_adj)
        combined_mse = mean_squared_error(y_test_reg_adj, combined_predictions)
        print(f"Combined Model MSE: {combined_mse}")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_reg_adj.values, label="Actual Weekly Prices", color="blue")
        plt.plot(regression_predictions_adj, label="Regression Predictions", color="green")
        plt.plot(lstm_predictions_adj, label="LSTM Predictions", color="red")
        plt.plot(combined_predictions, label="Combined Predictions", color="purple")
        plt.legend()
        plt.title("Model Predictions vs Actual Weekly Prices")
        plt.xlabel("Time (in weeks)")
        plt.ylabel("Price (in dollars)")
        plt.show()

        # Next Week Prediction using LSTM:
        full_data = pd.read_csv(file_name)
        full_data["timestamp"] = pd.to_datetime(full_data["timestamp"])
        full_data = full_data.sort_values("timestamp").set_index("timestamp")

        full_weekly = full_data.resample("W-FRI").last().dropna()
        full_close = full_weekly["close"].values.reshape(-1, 1)
        scaled_full = scaler_lstm.transform(full_close)

        sequence_length = 10
        last_sequence = scaled_full[-sequence_length:].reshape(1, sequence_length, 1)
        next_week_prediction_scaled = lstm_model.predict(last_sequence)
        next_week_prediction = scaler_lstm.inverse_transform(next_week_prediction_scaled)
        
        next_week_pred_value = next_week_prediction[0][0]
        print(f"Predicted next week's price: {next_week_pred_value}")

        # Return the prediction value so it can be used elsewhere
        return next_week_pred_value, full_weekly.index[-1]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    # Choose ticker accordingly
    symbol = "AAPL"
    next_week_pred_value, last_date = run_prediction(symbol)
    if next_week_pred_value is not None:
        next_week_date = last_date + pd.Timedelta(days=7)

        # Get DB creds from environment variables
        db_host = os.getenv("DB_HOST")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")

        mydb = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        mycursor = mydb.cursor()

        # Insert the prediction into the table
        sql = "INSERT INTO weekly_predictions (symbol, prediction_date, predicted_value) VALUES (%s, %s, %s)"
        val = (symbol, next_week_date.date(), float(next_week_pred_value))
        mycursor.execute(sql, val)
        mydb.commit()

        print("Prediction saved to MySQL database successfully!")

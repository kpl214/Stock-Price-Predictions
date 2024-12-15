import warnings
from api_fetch import fetch_stock_data
from preprocess_regression import preprocess_for_regression
from preprocess_lstm import preprocess_for_lstm
from train_regression import train_regression
from train_lstm import train_lstm
from train_xgboost import train_xgboost
from train_arima import train_arima
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

warnings.filterwarnings('ignore', category=FutureWarning)
load_dotenv()

def run_prediction(symbol):
    try:
        # Fetch Data
        file_name = fetch_stock_data(symbol, outputsize="full")

        # Preprocess data for regression and LSTM
        X_reg, y_reg, scaler_reg = preprocess_for_regression(file_name)
        X_lstm, y_lstm, scaler_lstm = preprocess_for_lstm(file_name, sequence_length=5)

        print("Shapes after preprocessing:")
        print("X_reg:", X_reg.shape, "y_reg:", y_reg.shape)
        print("X_lstm:", X_lstm.shape, "y_lstm:", y_lstm.shape)

        # Check for data sufficiency
        if len(X_reg) < 20:
            print("Warning: Insufficient data for regression model.")
        if len(X_lstm) < 20:
            print("Warning: Insufficient data for LSTM model.")

        # Split data for Regression
        train_size_reg = int(0.8 * len(X_reg))
        X_train_reg, X_test_reg = X_reg[:train_size_reg], X_reg[train_size_reg:]
        y_train_reg, y_test_reg = y_reg.iloc[:train_size_reg], y_reg.iloc[train_size_reg:]

        # Split data for LSTM
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
        lstm_model = train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, input_shape, epochs=50)

        # LSTM predictions (Test set)
        lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
        lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions_scaled)
        y_test_lstm_original = scaler_lstm.inverse_transform(y_test_lstm)
        lstm_mse = mean_squared_error(y_test_lstm_original, lstm_predictions)
        print(f"LSTM Model MSE (original scale): {lstm_mse}")

        # Train Regression
        regression_results = train_regression(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
        regression_predictions = regression_results["predictions"]
        regression_mse = mean_squared_error(y_test_reg, regression_predictions)
        print(f"Regression Model MSE: {regression_mse}")

        # Train XGBoost
        xgboost_results = train_xgboost(X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_jobs=-1)
        xgb_predictions = xgboost_results["predictions"]
        xgb_mse = xgboost_results["mse"]
        xgb_model = xgboost_results["model"]
        print(f"XGBoost Model MSE: {xgb_mse}")

        # SHAP for XGBoost
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_reg)

        # Read full data for ARIMA and next-week predictions
        full_data = pd.read_csv(file_name)
        full_data["timestamp"] = pd.to_datetime(full_data["timestamp"])
        full_data = full_data.sort_values("timestamp").set_index("timestamp")

        full_weekly = full_data.resample("W-FRI").last().dropna()
        full_close = full_weekly["close"].values.reshape(-1, 1)
        scaled_full = scaler_lstm.transform(full_close)

        # Train ARIMA on the full weekly data
        arima_results = train_arima(
            full_weekly["close"], 
            forecast_steps=1, 
            seasonal=True, 
            m=52,
            p_range=(0, 2), # range for p
            q_range=(0, 1), # range for q
            d_range=(0, 1) # range for d
        )
        arima_train_predictions = arima_results["train_predictions"]
        arima_forecast = arima_results["forecast"][0]
        arima_mse = arima_results["mse"]
        print(f"ARIMA Model MSE (in-sample): {arima_mse}")
        print(f"ARIMA Forecasted Next Week's Price: {arima_forecast}")

        # Align predictions for combining
        # Determine min_test_size
        min_test_size = min(len(y_test_reg), len(y_test_lstm), len(xgb_predictions), len(arima_train_predictions))

        # Truncate all series/arrays to the same length
        y_test_reg_adj = y_test_reg.iloc[:min_test_size]
        regression_predictions_adj = regression_predictions[:min_test_size]
        lstm_predictions_adj = lstm_predictions[:min_test_size].flatten()
        xgboost_predictions_adj = xgb_predictions[:min_test_size]
        arima_predictions_adj = arima_train_predictions.iloc[:min_test_size]

        # Convert all predictions to Series with the same index as y_test_reg_adj
        # This ensures proper alignment by date
        regression_series = pd.Series(regression_predictions_adj, index=y_test_reg_adj.index)
        lstm_series = pd.Series(lstm_predictions_adj, index=y_test_reg_adj.index)
        xgb_series = pd.Series(xgboost_predictions_adj, index=y_test_reg_adj.index)
        arima_series = pd.Series(arima_predictions_adj.values, index=y_test_reg_adj.index)

        # Combine all models equally
        lstm_weight, xgb_weight, arima_weight, regression_weight = 0.25, 0.25, 0.25, 0.25
        combined_series = (regression_weight * regression_series +
                           xgb_weight * xgb_series +
                           lstm_weight * lstm_series +
                           arima_weight * arima_series)
        combined_mse = mean_squared_error(y_test_reg_adj, combined_series)
        print(f"Combined Model MSE: {combined_mse}")

        # Plot results using indexed Series
        shap.summary_plot(shap_values, X_test_reg, feature_names=["SMA_20", "SMA_50", "daily_return", "volatility", "lag_close"])

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_reg_adj, label="Actual Weekly Prices", color="blue")
        plt.plot(regression_series, label="Regression Predictions", color="green")
        plt.plot(lstm_series, label="LSTM Predictions", color="red")
        plt.plot(xgb_series, label="XGBoost Predictions", color="orange")
        plt.plot(arima_series, label="ARIMA Predictions (Train)", color="black")
        plt.plot(combined_series, label="Combined Model Predictions", color="purple")
        plt.legend()
        plt.title("Model Predictions vs Actual Weekly Prices")
        plt.xlabel("Time")
        plt.ylabel("Price (in dollars)")
        plt.show()

        # Next Week Prediction combining LSTM and ARIMA
        sequence_length = 10
        last_sequence = scaled_full[-sequence_length:].reshape(1, sequence_length, 1)
        next_week_prediction_scaled = lstm_model.predict(last_sequence)
        next_week_prediction = scaler_lstm.inverse_transform(next_week_prediction_scaled)
        
        lstm_next = next_week_prediction[0][0]
        arima_next = arima_forecast
        combined_next_week = 0.5 * lstm_next + 0.5 * arima_next
        print(f"Predicted next week's price: {combined_next_week}")

        return combined_next_week, full_weekly.index[-1]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    symbol = "TSLA"
    combined_next_week, last_date = run_prediction(symbol)
    # Database insertion code can be uncommented when ready.
      # if next_week_pred_value is not None:
    #     next_week_date = last_date + pd.Timedelta(days=7)

    #     # Get DB creds from environment variables
    #     db_host = os.getenv("DB_HOST")
    #     db_user = os.getenv("DB_USER")
    #     db_password = os.getenv("DB_PASSWORD")
    #     db_name = os.getenv("DB_NAME")

    #     mydb = mysql.connector.connect(
    #         host=db_host,
    #         user=db_user,
    #         password=db_password,
    #         database=db_name
    #     )
    #     mycursor = mydb.cursor()

    #     # Insert the prediction into the table
    #     sql = "INSERT INTO weekly_predictions (symbol, prediction_date, predicted_value) VALUES (%s, %s, %s)"
    #     val = (symbol, next_week_date.date(), float(next_week_pred_value))
    #     mycursor.execute(sql, val)
    #     mydb.commit()

    #     print("Prediction saved to MySQL database successfully!")
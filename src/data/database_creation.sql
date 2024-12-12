CREATE DATABASE IF NOT EXISTS stock_predictions;
USE stock_predictions;

CREATE TABLE IF NOT EXISTS weekly_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10),
    prediction_date DATE,
    predicted_value FLOAT
);

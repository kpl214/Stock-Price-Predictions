# Stock-Price-Predictions

This project predicts stock prices using machine learning models, including LSTMs, XGBoost, and Linear Regression. It fetches stock data from the Alpha Vantage API, preprocesses it, trains models, and will eventually visualize predictions using a Streamlit dashboard.

Features

	•	Fetches stock data automatically from the Alpha Vantage API.
	•	Implements multiple models for stock price prediction:
	•	LSTM for time-series forecasting.
	•	XGBoost for tree-based regression.
	•	Linear Regression as a baseline model.
	•	Visualizes predictions and feature importance with a Streamlit app.

Setup

1. Clone the Repository

git clone https://github.com/yourusername/Stock-Price-Predictions.git
cd Stock-Price-Predictions

2. Install Dependencies

Create a virtual environment and install required libraries:

python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows

pip install -r requirements.txt

3. Set Up Environment Variables

Create a .env file in the root folder with the following:

ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

4. Run the Code

To train models and make predictions:

python main.py

To visualize results with Streamlit:

streamlit run dashboard.py


Future Work

	•	Store predictions in a cloud database (e.g., MySQL/Amazon RDS).
	•	Tune models for better performance.
	•	Add more models and external datasets.

Feel free to adapt or improve this README for your project!

import os
from dotenv import load_dotenv
import requests

# Load API key from .env file
load_dotenv()
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_stock_data(symbol, function="TIME_SERIES_DAILY"):
    
    # Fetch stock data from Alpha Vantage and save it as a CSV file.

    # Args:
    #     symbol (str): Stock symbol (e.g., "AAPL").
    #     function (str): Alpha Vantage function (default: "TIME_SERIES_DAILY").

    # Returns:
    #     str: The file name of the saved CSV.
    
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&datatype=csv"

    response = requests.get(url)
    if response.status_code == 200:
        file_name = f"{symbol}_stock_data.csv"
        with open(file_name, "wb") as file:
            file.write(response.content)
        print("Data fetched successfully.")
        return file_name
    else:
        print("Data fetch unsuccessfully.")

# data = pd.read_csv("stock_data.csv")
# data["timestamp"] = pd.to_datetime(data["timestamp"])
# data = data.sort_values("timestamp")

# plt.figure(figsize=(10,5))
# plt.plot(data["timestamp"], data["close"], label="Close Price")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.title(f"{symbol} Stock Price")
# plt.legend()
# plt.show()
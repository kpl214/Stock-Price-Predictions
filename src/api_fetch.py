import os
from dotenv import load_dotenv
import requests
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

symbol = "AAPL"
function = "TIME_SERIES_DAILY"
url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&datatype=csv"

response = requests.get(url)
if response.status_code == 200:
    with open("stock_data.csv", "wb") as file:
        file.write(response.content)
else:
    print("Failure to fetch data.")

data = pd.read_csv("stock_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values("timestamp")

# plt.figure(figsize=(10,5))
# plt.plot(data["timestamp"], data["close"], label="Close Price")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.title(f"{symbol} Stock Price")
# plt.legend()
# plt.show()
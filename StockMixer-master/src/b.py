import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np


tickers = ["0700.HK"]
company_names = ["腾讯控股"]

start_date = datetime(2022, 1, 1).strftime("%Y-%m-%d")
end_date = datetime(2025, 1, 1).strftime("%Y-%m-%d")

data = yf.download(tickers[0], start=start_date, end=end_date)
print(data)




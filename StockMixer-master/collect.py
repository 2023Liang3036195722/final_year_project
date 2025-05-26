import time
import random
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

tickers = ["0700.HK", "9988.HK", "1398.HK", "0941.HK", "0939.HK", "0005.HK", "3988.HK", "1810.HK", "3690.HK", "9618.HK"]
company_names = ["腾讯控股", "阿里巴巴 - W", "工商银行", "中国移动", "建设银行", "汇丰控股", "中银香港", "小米集团 - W", "美团 - W", "京东集团 - SW"]

start_date = datetime(2022, 1, 1).strftime("%Y-%m-%d")
end_date = datetime(2025, 4, 1).strftime("%Y-%m-%d")


def handle_nan(data):
    for col in data.columns:
        col_data = data[col]
        nan_indices = np.isnan(col_data)
        while nan_indices.any():
            index = np.where(nan_indices)[0][0]
            print('空值：'+str(index))
            left_index = index - 1
            right_index = index + 1
            while left_index >= 0 and np.isnan(col_data[left_index]):
                left_index -= 1
            while right_index < len(col_data) and np.isnan(col_data[right_index]):
                right_index += 1
            if left_index >= 0 and right_index < len(col_data):
                col_data[index] = (col_data[left_index] + col_data[right_index]) / 2
            elif left_index >= 0:
                col_data[index] = col_data[left_index]
            elif right_index < len(col_data):
                col_data[index] = col_data[right_index]
            nan_indices = np.isnan(col_data)
    return data

num_features = 5
first = True

for i, ticker in enumerate(tickers):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = handle_nan(data)
    num_days = len(data)
    stock_data = np.zeros((1, num_days, num_features))
    stock_data[0, :, 0] = data['High'].squeeze()
    stock_data[0, :, 1] = data['Low'].squeeze()
    stock_data[0, :, 2] = data['Open'].squeeze()
    stock_data[0, :, 3] = data['Volume'].squeeze()
    stock_data[0, :, 4] = data['Close'].squeeze()

    if first:
        np.save('stock_data.npy', stock_data)
        first = False
    else:
        existing_data = np.load('stock_data.npy')
        new_data = np.concatenate((existing_data, stock_data), axis=0)
        np.save('stock_data.npy', new_data)

    if i < len(tickers) - 1:
        time.sleep(0.5 + random.random())

loaded_stock_data = np.load('stock_data.npy')
# 检查是否有空值
has_nan = np.isnan(loaded_stock_data).any()
if has_nan:
    print("数据中存在空值，空值的行和列信息如下：")
    nan_indices = np.argwhere(np.isnan(loaded_stock_data))
    for index in nan_indices:
        stock_idx, day_idx, feature_idx = index
        print(f"股票索引: {stock_idx}, 日期索引: {day_idx}, 特征索引: {feature_idx}")
else:
    print("数据中不存在空值")


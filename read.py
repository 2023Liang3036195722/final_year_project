import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_parquet('data/stocks/0700.parquet')

print(df.head())
print(df.info())
print(df.describe())
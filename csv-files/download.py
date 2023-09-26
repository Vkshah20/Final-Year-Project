import yfinance as yf
import os

folder = 'csv-files'

file = open(os.path.join(folder, "coin_list.txt"))

for coin in file.readlines():
    try:
        ticker = yf.Ticker(coin.strip())
        df = ticker.history(period="max")
        df.to_csv(os.path.join(folder, coin.strip() + '.csv'))
    except Exception as e:
        print(e)
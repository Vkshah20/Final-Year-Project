from keras.models import load_model
import pandas as pd
import numpy as np
def multidense(start,end):
    model = load_model('./model/multi_setp_dense_k_100.h5')
    df = pd.read_csv("./csv-files/BTC-USD.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= start]
    df = df[df['Date'] <= end]
    df = df.reset_index(drop=True)
    indexes = df.index.values
    endindex = max(indexes)
    dc = pd.DataFrame(df[['Open', 'Close', 'High', 'Low']])
    # EG = df.iloc[0:100]
    predict = dc.iloc[endindex-100 : endindex]
    date = pd.DataFrame(df[['Date']])
    date = date.iloc[endindex:]
    expecteddf = pd.DataFrame(df[['Date', 'Close']])
    expecteddf['Type'] = 'Observed'
    tm = [3667.54092308, 3671.81879714, 3769.26483999, 3557.91619642]
    ts = [3962.52179229, 3963.50740991, 4095.59464515, 3805.53750232]
    predict = (predict - tm) / ts
    predict = np.expand_dims(predict, axis = 0)
    x = model.predict(predict)
    x = (x[0] * ts[1]) + tm[1]
    dataframe = pd.DataFrame(x, columns=['Close'])
    dataframe1 = pd.concat([date.reset_index(drop=True), dataframe], axis=1)
    dataframe1['Type'] = 'Predicted'
    print("\n\nAs this Multi Step Dense Model has one output might be possible that you undable to see on screen so output is displayed in terminal.")
    print(dataframe1)
    real = pd.concat([expecteddf, dataframe1]).reset_index(drop=True)
    return real
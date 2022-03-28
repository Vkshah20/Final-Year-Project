from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pickle
import multistepdense as i
import conventionalmodel as p
import lstm1 as t
import lstm3 as m
import lstm4 as l
import bilstm2 as v
import bilstm1 as q

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

control = dbc.Card([
    html.Div(
        [
            dbc.Label("Crypto-Currency :-"),
            dcc.Dropdown({
                'BTC-USD.csv': 'BTC',
                'ETC-USD.csv': 'ETC',
                'CRO-USD.csv': 'CRO'
            },
                value="BTC-USD.csv", id="demo-dropdown"
            ),
            dbc.Label("Model"),
            dcc.Dropdown({
                'arima.pickle': 'ARIMA',
                'arma.pickle' : 'ARMA',
                'MSDM' : 'Multi Step Dense Model',
                'CON' : 'Convention Model',
                'LSTM1' : 'LSTM taking past 100 days Data and predicting the next one day data',
                'LSTM3' : 'LSTM predicting only closing values',
                'LSTM4': 'LSTM predicting all values',
                'BILSTM2' : 'BILSTM predicting all values',
                'BILSTM1' : 'BILSTM predicting only closing values'
            },
                value="arima.pickle", id="model", optionHeight=55,
            ),
        ]
    ),
    html.Br(),
    html.Div(
        [
            dbc.Label("Date :-"),
            html.Br(),
            dcc.DatePickerRange(id='demo-date', start_date='2018-09-17', end_date='2019-01-24')
        ]
    ),
    html.Div(id='dd-output-container'),
], body=True)

app.layout = dbc.Container(
    [
        html.H1("Cryptocurrency Chart"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(control, md=4),
                dbc.Col(dcc.Graph(id="finance"), md=8),
            ],
            align="center",
        ),
        dbc.Row(dcc.Graph(id="candle"), align='center')
    ],
    fluid=True,
)


@app.callback(
    [Output('dd-output-container', 'children'), Output('finance', 'figure'), Output('candle', 'figure')],
    [Input('demo-dropdown', 'value'), Input('demo-date', 'start_date'), Input('demo-date', 'end_date'),Input('model','value')]
)
def update_output(value, start_date, end_date,model):
    sat = 'You have selected ' + value + ' from ' + start_date + ' to ' + end_date + ' using ' + model + ' Model.'
    df = pd.read_csv('./csv-files/' + value)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    if model=='arima.pickle' or model=='arma.pickle':
        df1 = df[['Date', 'Close']]
        df1['Type'] = 'Observed'
        index = df1.index.values
        # loading arima pickle file
        file_open = open("./model/"+model, "rb")
        arima_model = pickle.load(file_open)
        df3 = arima_model.predict(min(index), max(index))
        dfdate = df1['Date']
        df2 = pd.DataFrame({'Date': dfdate, 'Close': df3, 'Type': 'Predicted'})
        # combining two dataFrame
        result = pd.concat([df1, df2])

    elif model=='LSTM4':
        result=l.lstm(start_date,end_date)

    elif model=='BILSTM2':
        result=v.bilstm(start_date,end_date)

    elif model=='BILSTM1':
        result=q.bilstm(start_date,end_date)

    elif model=='LSTM3':
        result=m.lstm(start_date,end_date)

    elif model=='LSTM1':
        result=t.lstm(start_date,end_date)

    elif model=='CON':
        result=p.convenmodel(start_date,end_date)

    elif model=='MSDM':
        result=i.multidense(start_date,end_date)
    
    fig = px.line(data_frame=result, x='Date', y='Close', color='Type')
    fig2 = go.Figure(
        data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    return sat, fig, fig2


if __name__ == '__main__':
    app.run_server(debug=True)

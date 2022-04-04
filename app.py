from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pickle
import feature4 as v
import feature1 as q

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
                'arma.pickle': 'ARMA',
                'bilstm_model_k_100_o_110_feature_1.h5': 'BILSTM Model predicting only closing price',
                'conv_bilstm_k_100_o_110_feature_1.h5': 'Conventional-BILSTM Model predicting only closing price',
                'conv_lstm_k_100_o_110_feature_1.h5': 'Conventional-LSTM Model predicting only closing price',
                'conv_model_k_100_o_110_feature_1.h5': 'Conventional Model predicting only closing price',
                'dense_k_100_o_110_feature_1.h5': 'Dense Model predicting only closing price',
                'lstm_bilstm_k_100_o_110_feature_1.h5': 'LSTM-BILSTM Model predicting only closing price',
                'lstm_model_k_100_o_110_feature_1.h5': 'LSTM Model predicting only closing price',
                'multi_step_dense_k_100_o_110_feature_1.h5': 'Multi Step Dense Model predicting only closing price',
                'bilstm_model_k_100_o_110_feature_4.h5': 'BILSTM Model predicting all four feature',
                'conv_bilstm_k_100_o_110_feature_4.h5': 'Conventional-BILSTM Model predicting all four feature',
                'conv_lstm_k_100_o_110_feature_4.h5': 'Conventional-LSTM Model predicting all four feature',
                'conv_model_k_100_o_110_feature_4.h5': 'Conventional Model predicting all four feature',
                'dense_k_100_o_110_feature_4.h5': 'Dense Model predicting all four feature',
                'lstm_bilstm_k_100_o_110_feature_4.h5': 'LSTM-BILSTM Model predicting all four feature',
                'lstm_model_k_100_o_110_feature_4.h5': 'LSTM Model predicting all four feature',
                'multi_step_dense_k_100_o_110_feature_4.h5': 'Multi Step Dense Model predicting all four feature',
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
    feature1=list(("bilstm_model_k_100_o_110_feature_1.h5", "conv_bilstm_k_100_o_110_feature_1.h5", "conv_lstm_k_100_o_110_feature_1.h5", "conv_model_k_100_o_110_feature_1.h5", "dense_k_100_o_110_feature_1.h5", "lstm_bilstm_k_100_o_110_feature_1.h5", "lstm_model_k_100_o_110_feature_1.h5", "multi_step_dense_k_100_o_110_feature_1.h5"))
    feature4=list(("bilstm_model_k_100_o_110_feature_4.h5", "conv_bilstm_k_100_o_110_feature_4.h5", "conv_lstm_k_100_o_110_feature_4.h5", "conv_model_k_100_o_110_feature_4.h5", "dense_k_100_o_110_feature_4.h5", "lstm_bilstm_k_100_o_110_feature_4.h5", "lstm_model_k_100_o_110_feature_4.h5", "multi_step_dense_k_100_o_110_feature_4.h5"))
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
        file_open = open("./finalmodel/"+model, "rb")
        arima_model = pickle.load(file_open)
        df3 = arima_model.predict(min(index), max(index))
        dfdate = df1['Date']
        df2 = pd.DataFrame({'Date': dfdate, 'Close': df3, 'Type': 'Predicted'})
        # combining two dataFrame
        result = pd.concat([df1, df2])

    elif model in feature4:
        result=v.predict4(start_date, end_date, model)

    elif model in feature1:
        result=q.predict1(start_date, end_date, model)

    fig = px.line(data_frame=result, x='Date', y='Close', color='Type')
    fig2 = go.Figure(
        data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    return sat, fig, fig2


if __name__ == '__main__':
    app.run_server(debug=True)

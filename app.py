from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

control=dbc.Card([
        html.Div(
            [
                dbc.Label("Crypto-Currency :-"),
                dcc.Dropdown({
                        'BTC-USD.csv':'BTC',
                        'ETC-USD.csv':'ETC',
                        'CRO-USD.csv':'CRO'
                },
                value="BTC-USD.csv",id="demo-dropdown"
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Date :-"),
                html.Br(),
                dcc.DatePickerRange(id='demo-date',start_date='2021/6/1',end_date='2022/2/23')
            ]
        ),
        html.Div(id='dd-output-container'),
],body=True)

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
        dbc.Row(dcc.Graph(id="candle"),align='center')
    ],
    fluid=True,
)

@app.callback(
    [Output('dd-output-container', 'children'),Output('finance', 'figure'),Output('candle', 'figure')],
    [Input('demo-dropdown', 'value'),Input('demo-date', 'start_date'),Input('demo-date', 'end_date')]
)
def update_output(value,start_date,end_date):
    sat='You have selected '+ value+ ' from '+start_date+' to '+end_date
    df=pd.read_csv('./csv-files/'+value)
    df=df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    fig=px.line(data_frame=df,x='Date',y='Close')
    fig2=go.Figure(data=[go.Candlestick(x=df['Date'],open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'])])
    return sat,fig,fig2

if __name__ == '__main__':
    app.run_server(debug=True)
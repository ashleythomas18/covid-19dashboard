import requests
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from sklearn.linear_model import LinearRegression

# Fetch COVID-19 data
def fetch_data():
    url = "https://disease.sh/v3/covid-19/historical/all?lastdays=all"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Convert nested JSON into DataFrame
        df = pd.DataFrame({
            "Date": list(data['cases'].keys()),
            "Confirmed": list(data['cases'].values()),
            "Deaths": list(data['deaths'].values()),
            "Recovered": list(data['recovered'].values())
        })
        return df
    else:
        raise Exception("Failed to fetch data from API.")

# Preprocess the data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
    return df

# Predict future cases (simple linear regression model)
def predict_cases(df):
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days 
    model = LinearRegression()
    X = df['Days'].values.reshape(-1, 1)
    y = df['Confirmed'].values
    model.fit(X, y)
    future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + 31).reshape(-1, 1)
    predictions = model.predict(future_days)
    return pd.DataFrame({'Days': future_days.flatten(), 'Predicted_Confirmed': predictions})

# Fetch and preprocess the data
df = fetch_data()
df = preprocess_data(df)
predictions = predict_cases(df)

latest_metrics = {
    "Confirmed": df['Confirmed'].iloc[-1],
    "Deaths": df['Deaths'].iloc[-1],
    "Recovered": df['Recovered'].iloc[-1],
    "Active": df['Active'].iloc[-1],
}

# Create the Dash app
app = Dash(__name__)
app.layout = html.Div(
    style={"display": "flex", "backgroundColor": "#f7f7f7", "minHeight": "100vh"},
    children=[
        html.Div(
            style={"width": "20%", "padding": "20px", "backgroundColor": "#1e3d59", "color": "white"},
            children=[
                html.H2("COVID-19 Dashboard", style={"textAlign": "center", "marginBottom": "30px"}),
                dcc.Dropdown(
                    id='data-type',
                    options=[
                        {'label': 'Confirmed Cases', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'},
                        {'label': 'Active Cases', 'value': 'Active'}
                    ],
                    value='Confirmed',
                    style={'marginBottom': '20px', 'color': 'black'},
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=df['Date'].min(),
                    end_date=df['Date'].max(),
                    display_format='YYYY-MM-DD',
                    style={'marginBottom': '20px'}
                ),
                html.H4("Key Metrics", style={"marginTop": "20px"}),
                html.Div([
                    html.Div(
                        style={"padding": "10px", "marginBottom": "10px", "backgroundColor": "#3c6e71"},
                        children=f"Confirmed: {latest_metrics['Confirmed']:,}"
                    ),
                    html.Div(
                        style={"padding": "10px", "marginBottom": "10px", "backgroundColor": "#284b63"},
                        children=f"Deaths: {latest_metrics['Deaths']:,}"
                    ),
                    html.Div(
                        style={"padding": "10px", "marginBottom": "10px", "backgroundColor": "#4a7c8b"},
                        children=f"Recovered: {latest_metrics['Recovered']:,}"
                    ),
                    html.Div(
                        style={"padding": "10px", "marginBottom": "10px", "backgroundColor": "#507e84"},
                        children=f"Active: {latest_metrics['Active']:,}"
                    ),
                ])
            ],
        ),
        html.Div(
            style={"width": "80%", "padding": "20px"},
            children=[
                dcc.Graph(id='covid-graph'),
                html.H2("Future Predictions for Confirmed Cases", style={'textAlign': 'center', "marginTop": "30px"}),
                dcc.Graph(
                    id='prediction-graph', 
                    figure=px.line(
                        predictions, 
                        x='Days', 
                        y='Predicted_Confirmed', 
                        title='Predicted COVID-19 Cases for the Next 30 Days',
                        labels={'Days': 'Days (from start)', 'Predicted_Confirmed': 'Predicted Cases'}
                    ).update_layout(template="plotly_dark")
                ),
            ]
        )
    ]
)

# Callbacks for updating graphs
@app.callback(
    Output('covid-graph', 'figure'),
    [Input('data-type', 'value'), Input('date-picker', 'start_date'), Input('date-picker', 'end_date')]
)
def update_graph(data_type, start_date, end_date):
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    fig = px.line(
        filtered_df, 
        x='Date', 
        y=data_type, 
        title=f'COVID-19 {data_type} Over Time',
        labels={'Date': 'Date', data_type: data_type},
        template="plotly_white"
    ).update_layout(transition_duration=500)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
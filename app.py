import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import timedelta

st.set_page_config(page_title="EV Charging Optimization", layout="wide")

st.title("🔋 EV Charging & Solar PV Generation Dashboard")

# Sidebar for file upload
st.sidebar.header("📂 Upload Data")

ev_file = st.sidebar.file_uploader("Upload EV Charging Station file (.csv or .xlsx)", type=['csv', 'xlsx'])
solar_file = st.sidebar.file_uploader("Upload Solar PV Generation file (.csv)", type='csv')

# Read EV file
def read_ev_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)

# Process EV file
if ev_file:
    ev_data = read_ev_file(ev_file)
    st.subheader("EV Charging Station Data")
    st.write(ev_data.head())

    # Clean and convert numeric data
    for col in ev_data.columns:
        if ev_data[col].dtype == object and ev_data[col].str.contains('kW', na=False).any():
            ev_data[col] = ev_data[col].str.replace('kW', '', regex=False).astype(float)

    st.success("EV Charging data cleaned successfully!")

# Process Solar PV file
if solar_file:
    solar_df = pd.read_csv(solar_file)
    st.subheader("Solar PV Generation Data")
    st.write(solar_df.head())

    # Parse datetime and sort
    solar_df['DATE_TIME'] = pd.to_datetime(solar_df['DATE_TIME'], dayfirst=True)
    solar_df = solar_df.sort_values('DATE_TIME')

    # Plot line chart
    st.subheader("📈 Solar Generation Over Time")
    fig = px.line(solar_df, x='DATE_TIME', y='DC_POWER', title='DC Power Generation')
    st.plotly_chart(fig, use_container_width=True)

    # Prepare for LSTM
    df = solar_df[['DATE_TIME', 'DC_POWER']].copy()
    df.set_index('DATE_TIME', inplace=True)

    # Resample hourly and normalize
    df_hourly = df.resample('1H').mean().fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_hourly)

    # Prepare sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    SEQ_LENGTH = 24
    X, y = create_sequences(scaled, SEQ_LENGTH)

    # Split and reshape
    X_train, y_train = X[:-24], y[:-24]
    X_forecast = X[-1:]
    y_true = y[-24:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Forecast
    forecast = model.predict(X_forecast)
    forecast_inv = scaler.inverse_transform(forecast)

    # Show results
    st.subheader("🔮 Forecasted Next Hour Solar Generation (kW)")
    st.metric("Prediction", f"{forecast_inv[0][0]:.2f} kW")

    # Past 24 hrs and prediction chart
    st.subheader("📊 Recent DC Power and Prediction")
    recent_df = df_hourly[-24:].copy()
    future_time = recent_df.index[-1] + timedelta(hours=1)
    recent_df.loc[future_time] = forecast_inv[0][0]

    fig2 = px.line(recent_df, title="Recent and Forecasted DC Power")
    st.plotly_chart(fig2, use_container_width=True)

if not ev_file or not solar_file:
    st.info("👈 Upload both EV and Solar PV datasets to start analysis.")
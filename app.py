import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from datetime import timedelta, datetime

st.set_page_config(page_title="EV Charging Optimization", layout="wide")
st.title("🔋 EV Charging & Solar PV Generation Dashboard")

# Sidebar file uploads
st.sidebar.header("📂 Upload Data")
ev_file = st.sidebar.file_uploader("Upload EV Charging Station file (.csv or .xlsx)", type=['csv', 'xlsx'])
solar_file = st.sidebar.file_uploader("Upload Solar PV Generation file (.csv)", type='csv')

# Read EV file
def read_ev_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)

if ev_file:
    ev_data = read_ev_file(ev_file)
    st.subheader("EV Charging Station Data")
    st.write(ev_data.head())

    for col in ev_data.columns:
        if ev_data[col].dtype == object and ev_data[col].str.contains('kW', na=False).any():
            ev_data[col] = ev_data[col].str.replace('kW', '', regex=False).astype(float)

    st.success("✅ EV Charging data cleaned successfully!")

if solar_file:
    solar_df = pd.read_csv(solar_file)
    st.subheader("Solar PV Generation Data")
    st.write(solar_df.head())

    solar_df['DATE_TIME'] = pd.to_datetime(solar_df['DATE_TIME'], dayfirst=True)
    solar_df = solar_df.sort_values('DATE_TIME')

    # Line chart
    st.subheader("📈 Solar Generation Over Time")
    fig = px.line(solar_df, x='DATE_TIME', y='DC_POWER', title='DC Power Generation')
    st.plotly_chart(fig, use_container_width=True)

    # Prepare for LSTM
    df = solar_df[['DATE_TIME', 'DC_POWER']].copy()
    df.set_index('DATE_TIME', inplace=True)
    df_hourly = df.resample('1H').mean().fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_hourly)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    SEQ_LENGTH = 24
    X, y = create_sequences(scaled, SEQ_LENGTH)
    X_train, y_train = X[:-24], y[:-24]

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, 1)),
        Dense(1),
        Activation('relu')  # Ensures output is non-negative
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Time Picker
    st.subheader("🕒 Select Forecast Start Time")

    min_time = df_hourly.index.min().to_pydatetime()
    max_time = df_hourly.index.max().to_pydatetime()

    selected_date = st.date_input("Choose a date", value=min_time.date(), min_value=min_time.date(), max_value=max_time.date())
    selected_time = st.time_input("Choose a time", value=min_time.time())

    forecast_start = pd.Timestamp(datetime.combine(selected_date, selected_time))

    if forecast_start not in df_hourly.index:
        st.warning("⚠️ Selected time not available in data. Using closest available time.")
        forecast_start = df_hourly.index[df_hourly.index.get_indexer([forecast_start], method='nearest')[0]]

    start_idx = df_hourly.index.get_loc(forecast_start)

    if start_idx + 24 > len(df_hourly):
        st.error("❌ Not enough data after the selected time to perform forecasting.")
    else:
        seed_seq = scaled[start_idx:start_idx + 24].reshape(1, 24, 1)

        # Forecast next 6 hours
        forecast_values = []
        for _ in range(6):
            pred = model.predict(seed_seq, verbose=0)[0]
            forecast_values.append(pred)
            seed_seq = np.append(seed_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        forecast_inv = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))
        forecast_inv = np.clip(forecast_inv, 0, None)

        forecast_times = [forecast_start + timedelta(hours=i+1) for i in range(6)]

        st.subheader("🔮 Forecasted Solar Generation (Next 6 Hours)")
        forecast_df = pd.DataFrame({
            'Time': forecast_times,
            'Forecasted_DC_Power': forecast_inv.flatten()
        })
        st.write(forecast_df)

        fig2 = px.line(forecast_df, x='Time', y='Forecasted_DC_Power', title="Forecasted Solar Generation (Next 6 Hours)")
        st.plotly_chart(fig2, use_container_width=True)

        # Optimization
        if ev_file:
            st.subheader("⚡ EV Charging Optimization Suggestion")
            ev_demand = ev_data.select_dtypes(include=[np.number]).sum(axis=1).tail(6).values
            solar_supply = forecast_inv.flatten()

            optimization = []
            for i in range(6):
                if solar_supply[i] >= ev_demand[i] if i < len(ev_demand) else 0:
                    optimization.append("✔️ Charge Now")
                else:
                    optimization.append("⏸️ Delay Charging")

            # Buffer check logic
            buffer_margin = 1.10  # 10% margin
            buffered_plan = []
            for i in range(6):
                required = ev_demand[i] if i < len(ev_demand) else 0
                if solar_supply[i] >= required * buffer_margin:
                    buffered_plan.append("✅ Sufficient with buffer")
                elif solar_supply[i] >= required:
                    buffered_plan.append("⚠️ Just Enough")
                else:
                    buffered_plan.append("⛔ Insufficient")

            forecast_df["EV Demand (kW)"] = np.append(ev_demand, [np.nan] * (6 - len(ev_demand)))
            forecast_df["Charging Plan"] = optimization
            forecast_df["Buffer Check"] = buffered_plan

            st.dataframe(forecast_df)

            # Overlay chart: Solar vs EV demand
            comparison_df = forecast_df.dropna().copy()
            fig3 = px.line(comparison_df, x='Time', y=['Forecasted_DC_Power', 'EV Demand (kW)'],
                           title="🔋 Forecasted Solar vs EV Demand (Next 6 Hours)",
                           labels={"value": "Power (kW)", "variable": "Metric"})
            st.plotly_chart(fig3, use_container_width=True)

            # Download button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Forecast & Charging Plan CSV", data=csv,
                               file_name="ev_charging_plan.csv", mime='text/csv')

if not ev_file or not solar_file:
    st.info("👈 Upload both EV and Solar PV datasets to begin analysis.")

st.markdown("---")
st.header("🚀 Future Upgrades Coming Soon")
st.markdown("""
- ✅ **Dynamic EV load shifting** based on real-time solar
- 🧠 **AI-based optimal scheduling** (Reinforcement Learning)
- ☁️ **Cloud syncing and real-time alerts**
- 💡 **Battery storage prediction integration**
- 📊 **Daily/weekly optimization reports**
""")

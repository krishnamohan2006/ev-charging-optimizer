# ⚡ EV Charging Optimization using Solar Forecast

This Streamlit web app forecasts solar power generation using LSTM and helps visualize data for electric vehicle (EV) charging optimization.

## 📦 Files
- `app.py`: Main Streamlit app file.
- `requirements.txt`: List of dependencies.
- `README.md`: This documentation.

## 📈 Features
- Upload `.csv` or `.xlsx` EV Charging data.
- Upload `.csv` Solar PV data.
- Cleans and normalizes power data.
- Trains an LSTM neural network on solar DC power.
- Forecasts the next hour of solar energy.
- Displays interactive charts of past and future generation.

## 🔍 Model Used
- **LSTM (Long Short-Term Memory)** from Keras for time series prediction.
- Input: Last 24 hourly DC power values.
- Output: Forecast for next hour.

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/ev-charging-optimizer.git
cd ev-charging-optimizer

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📊 Output
- Line chart of hourly DC solar generation.
- Forecast of next hour's power.
- Metrics display and trend line chart.

## 📬 Sample Datasets
- Solar PV generation data with `DATE_TIME` and `DC_POWER` columns.
- EV Charging station data with `Capacity`, `Power Type`, etc.

## 📌 Notes
- Make sure datetime in solar data is in `dd-mm-yyyy HH:MM` format.
- Add your own dataset or use Kaggle-based data.

---
Made with ❤️ using Streamlit, Keras, and Plotly.
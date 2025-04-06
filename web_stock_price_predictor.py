import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")

# Input for Stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Fetching Stock Data
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
google_data = yf.download(stock, start, end)

# Load Model
model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

# Plotting Function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Calculate Moving Averages and Plot
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()

st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))
st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))
st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))
st.subheader('Original Close Price with MA for 100 and 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Data Preprocessing
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)
x_data, y_data = [], []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict using the Model
predictions = model.predict(x_data)

# Inverse Scaling
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Plot Results
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len+100:])

st.subheader("Original Values vs Predicted Values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data - Not Used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)

# Calculate Accuracy Metrics
mse = mean_squared_error(inv_y_test, inv_pre)
rmse = np.sqrt(mse)
mae = mean_absolute_error(inv_y_test, inv_pre)
r2 = r2_score(inv_y_test, inv_pre)

# Display Accuracy Metrics
st.subheader("Model Accuracy")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"R-Squared (R²): {r2:.4f}")

# Plot actual vs predicted graph for last 10 dates
st.subheader("Actual vs Predicted - Last 10 Days")
last_10 = ploting_data.tail(10)
fig10 = plt.figure(figsize=(15, 6))
plt.plot(last_10.index, last_10['original_test_data'], color='blue', label='Actual')
plt.plot(last_10.index, last_10['predictions'], color='orange', label='Predicted')
plt.legend()
st.pyplot(fig10)

# To run the app: python -m streamlit run web_stock_price_predictor.py
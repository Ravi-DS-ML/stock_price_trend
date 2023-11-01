import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Set the start and end dates
start_date = '2010-01-01'
end_date = '2019-12-31'

st.title("Stock Price Trend Prediction")

user_input = st.text_input("Enter Stock Ticker", "AAPL")

# Get the stock data based on the user's input
data = yf.download(user_input, start=start_date, end=end_date)

# Describe the data
st.header(f"Data for {user_input}")

st.write(data.describe())

st.subheader("Closing Price vs Time Chart")

fig = plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
st.pyplot(fig)

st.subheader("Moving Averages vs Time Chart")
fig_ma = plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Close'].rolling(100).mean(), label='MA100')
plt.plot(data['Close'].rolling(200).mean(), label='MA200')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig_ma)

# Splitting the data
train_data = pd.DataFrame(data['Close'][:int(len(data)) * 0.70])
test_data = pd.DataFrame(data['Close'][int(len(data) * 0.70):])

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.fit_transform(test_data)

# Load the pre-trained model
model = load_model('keras_model.h5')

# Testing part

# Prepare test data
past_100_days = train_data.tail(100)
final_df = pd.concat([past_100_days, test_data], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting the stock price

# Make predictions
y_predicted = model.predict(x_test)

# Inverse scaling to get actual prices
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the results
st.subheader("Prediction vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

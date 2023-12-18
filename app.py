import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# Set the start and end dates
start_date = '2010-01-01'
end_date = '2023-10-31'
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)
# Custom Streamlit styles
st.markdown(
    """
    <style>

    .stTitle {
        font-size: 32px;
        color: #333;
    }
    .stTextInput {
        width: 200px;
    }
    .stButton {
        background-color: #333;
        color: white;
        border-radius: 5px;
    }
    .stButton:hover {
        background-color: #555;
    }
    .stSubheader {
        font-size: 24px;
        color: #333;
    }
    .stWrite {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .metric-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stock Price Trend Prediction")
user_input = st.text_input("Enter Stock Ticker", "AAPL", key="user_input")

# Get the stock data based on the user's input
if user_input:
    data = yf.download(user_input, start=start_date, end=end_date)

    # Describe the data
    st.subheader(f"Data for {user_input}")
    st.write(data.describe(), key="data_description")

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
    train_data = pd.DataFrame(data['Close'][:int(len(data) * 0.70)])
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

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    # Display metrics with improved styling
    st.subheader("Prediction vs Original")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    st.subheader("Evaluation Metrics")
    st.markdown("<div class='metric-box'>"
                f"<h4>Mean Squared Error (MSE): {round(mse,3)}</h4>"
                "<p>MSE measures the average squared difference between the predicted and actual stock prices. Lower values indicate better predictions.</p>"
                f"<h4>Root Mean Squared Error (RMSE): {round(rmse,3)}</h4>"
                "<p>RMSE is the square root of MSE and provides an interpretable measure of prediction error. Lower RMSE indicates more accurate predictions.</p>"
                f"<h4>Mean Absolute Error (MAE): {round(mae,3)}</h4>"
                "<p>MAE measures the average absolute difference between predicted and actual prices. It is easy to understand, and lower values are desirable.</p>"
                f"<h4>R-squared (R2) Score: {round(r2,3)}</h4>"
                "<p>R2 measures the proportion of the variance in the stock prices that is predictable by your model. A higher R2 score indicates a better fit to the data (closer to 1).</p>"
                "</div>",
                unsafe_allow_html=True)

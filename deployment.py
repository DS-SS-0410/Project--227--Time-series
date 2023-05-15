import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Title of the app
st.title("Apple Stock Price Prediction App")

# Sidebar with options
st.sidebar.subheader("Choose the Options")

# Start and end dates
start_date = st.sidebar.date_input("Start date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 12, 31))

# Stock ticker
ticker = st.sidebar.text_input("Enter the Ticker symbol", "AAPL")

# Load the saved model
loaded_model = _unpickle("finalfitmodel2.pkl")

# Load the historical stock data
df = yf.download(ticker, start=start_date, end=end_date)

# Define the forecast period
forecast_period = st.sidebar.slider(
    "Select the forecast period (in days)", min_value=1, max_value=365, value=365
)

# Forecast button
if st.sidebar.button("Forecast"):
    try:
        # Train the model on the updated data
        model_fit = SARIMAX(
            endog=df["Close"],
            exog=None,
            order=loaded_model.specification['order'],
            seasonal_order=loaded_model.specification['seasonal_order'],
        )
        model_fit = model_fit.filter(loaded_model.params)

        # Make the forecast
        forecast = model_fit.forecast(steps=forecast_period)

        # Show the forecasted values
        st.write("Forecasted Values:")
        st.write(forecast)

        # Show the plot
        st.write("Historical Plot:")
        st.line_chart(df["Close"])
                
        # Show the plot
        st.write("Forecast Plot:")
        st.line_chart(forecast)

        

    except Exception as e:
        st.write("Oops! Something went wrong.")
        st.write(e)

# Show the historical data
st.subheader("Forecast Data")
st.write(df)


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Load all files
with open('linreg_model.pkl', 'rb') as file_1:
    model_lr = pickle.load(file_1)
with open('num_columns.pkl', 'rb') as file_2:
    scaler = pickle.load(file_2)

quantity_flow1 = pd.read_csv('quantity_flow1.csv')
quantity_flow1['week_start_date'] = pd.to_datetime(quantity_flow1['week_start_date'])
quantity_flow1.set_index('week_start_date', inplace=True)

# Define the forecasting function
def forecasting(month):
    # :param month: how many months to predict

    quantity_forecast = quantity_flow1.copy()
    window = 2
    for i in range(month):
        X = np.array(quantity_forecast[-window:].values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        new_idx = quantity_forecast.index[-1] + timedelta(weeks=1)
        quantity_forecast.loc[new_idx] = round(model_lr.predict(X_scaled)[0])
    return quantity_forecast.tail(month + window)

# Define the Streamlit app
def run():
    st.title("Quantity Sales Forecasting")
    
    # Input the number of months to forecast
    months_to_forecast = st.slider("How many weeks to forecast?", 1, 13, 3)

    # Predict button
    if st.button("Predict"):
        # Get the forecast
        quantity_forecast = forecasting(months_to_forecast)

        # Display the forecast plot
        fig, ax = plt.subplots(figsize=(20,5))
        ax.plot(quantity_flow1, color='red', label='Real Quantity')
        ax.plot(quantity_forecast, color='blue', label='Quantity Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity')
        ax.set_title(f"Quantity Forecast for the Next {months_to_forecast} Weeks")
        ax.legend()
        st.pyplot(fig)

        # Reset the index of quantity_forecast and rename the index column to "Date"
        quantity_forecast_table = quantity_forecast.reset_index().rename(columns={"index": "Date"})

        # Display the forecasted sales in a table
        st.write(quantity_forecast_table)

if __name__ == '__main__':
    run()
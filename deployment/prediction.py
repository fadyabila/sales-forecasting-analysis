import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

# Load All Files

with open('linreg_model.pkl', 'rb') as file_1:
  model_lr = pickle.load(file_1)
with open('num_columns.pkl', 'rb') as file_2:
  scaler = pickle.load(file_2)

def run():
    # Load the time series data
    quantity_flow1 = pd.read_csv('quantity_flow1.csv', index_col=0, parse_dates=True)
    
    # Function to Predict Quantity in Next n-weeks
    def forecasting(month):
        quantity_forecast = quantity_flow1.copy()
        window = 2
        for i in range(month):
            X = np.array(quantity_forecast[-window:].values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            new_idx = quantity_forecast.index[-1] + timedelta(weeks=1)
            quantity_forecast[new_idx] = round(model_lr.predict(X_scaled)[0])
        return quantity_forecast
    
    # Set up the Streamlit app
    st.title("Sales Forecasting")
    
    # Create a sidebar for user input
    months = st.sidebar.slider("Select the number of weeks to forecast", 1, 24, 12)
    
    # Generate the forecast and plot the results
    quantity_forecast = forecasting(months)
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(quantity_forecast, color='blue', label='forecast')
    ax.plot(quantity_flow1, color='red', label='real')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    run()
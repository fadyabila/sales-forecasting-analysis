import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page : ', ('EDA', 'Quantity Sales Forecasting'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()
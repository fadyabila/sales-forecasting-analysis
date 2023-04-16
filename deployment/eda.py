import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Melebarkan visualisasi untuk memaksimalkan browser
st.set_page_config(
    page_title='Sales Forecasting Analysis in Paragon Corp',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Membuat title
    st.title('Beauty Products Warehouse Sales Forecasting Analysis in Paragon Corp')
    st.write('### by Fadya Ulya Salsabila')

    # Menambahkan Gambar
    image = Image.open('paragon.png')
    st.image(image, caption='Paragon Corp')

    # Menambahkan Deskripsi
    st.write('## Background')
    st.write("""
    Paragon is an Indonesian beauty/cosmetics company with the goal of creating the greater good for society through innovation. Products from Paragon are Wardah, Kahf, Make Over, and Emina. 
    Targeting the middle class segment for cosmetics is attractive because customers are relatively insensitive to price. Like Paragon, it is a local organization targeting middle class customers in Indonesia.

    Warehouses within a company must be adapted to the conditions and intensity of production in the industry. 
    Meanwhile, warehousing is an important line in the trading business, because there are existing industrial and production goods, such as receiving raw materials from suppliers, handling goods, sending goods to their destination.
    The warehouse management system is very important for business continuity, because the warehouse is directly related to sales. 
    When the warehouse inventory does not match sales, it will have an impact on losses, either due to failed sales or too much inventory available in the warehouse. 
    A warehouse management system whose main purpose is to control all processes that occur in it, such as receiving, storing, processing customer orders, taking orders, checking and packing and shipping. 
    With a warehouse management system, we can better control the process of movement and storage, more optimal use of space in the warehouse, increase the effectiveness of the receiving and shipping process and know the amount of stock more accurately from time to time.

    Therefore, in this analysis and modeling, this will predict the number of sales of Paragon's warehouse stock using the `sample_dataset_timeseries_noarea.csv` dataset. 
    This prediction will use Regression Model with Time Series and Forecasting Analysis.""")

    st.write('## Dataset')
    st.write("""
    The dataset is from Paragon dataset. 
    This dataset contains `102733 rows` and `5 columns`, such as:

    1. `week_number`: contained information about week of specific product sold, (2021-52 to 2023-14).
    2. `week_start_date`: contained information about week start date of specific product sold.
    3. `week_end_date`: contained information about week end date of specific product sold.
    4. `product_item`: contained information about product item/product code (Variabel Bebas).
    5. `quantity`: contained information about quantity of product in respective week.""")

    # Membuat Garis Lurus
    st.markdown('---')

    # Membuat Sub Headrer
    st.subheader('EDA for Sales Forecasting')

    # Magic Syntax
    st.write(
    ' On this page, the author will do a simple exploration.'
    ' The dataset used is the Quantity Sales dataset.'
    ' This dataset comes from Paragon Corp.')

    # Show DataFrame
    df1 = pd.read_csv('sample_dataset_timeseries_noarea.csv')
    st.dataframe(df1)

    # Membuat Barplot
    st.write('#### Product Item Plot')
    fig = plt.figure(figsize=(10,7))
    sns.countplot(x='product_item', data=df1, palette="PuRd")
    st.pyplot(fig)
    st.write(
    ' There are 2309 different product items.' 
    ' Where the product item appears the most, such as 67 times and the least appears is 1 time.')

    # Mengelompokkan data
    # Convert date column to a datetime object
    df1["week_start_date"] = pd.to_datetime(df1["week_start_date"])

    # Group by week_start_date and sum quantity
    st.write('#### Quantity Grouped by Weeks')
    quantity_flow = df1.groupby(pd.Grouper(key="week_start_date", freq="W-MON"))["quantity"].sum().reset_index()
    quantity_flow = quantity_flow.rename(columns={"week_start_date": "week_start_date"})
    st.write(
    ' The column for the number of products sold (`quantity`) is grouped by `week_start_date`, intended to find out how many product stocks have been sold each week.' 
    ' Based on the results above, there are 67 sales weeks with a different number of products each week.')

    # Plot the quantity_flow DataFrame
    st.write('#### Quantity Flow')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(quantity_flow["week_start_date"], quantity_flow["quantity"], color="blue", linestyle="dashed")
    ax.set_title("Quantity Flow")
    ax.set_xlabel("Week Start Date")
    ax.set_ylabel("Quantity")
    ax.grid(True)
    st.pyplot(fig)
    st.write("""
    The visualization of the trend chart above shows that warehouse stock sales fluctuate. 
    Especially in May 2022 it shows that sales have fallen dramatically from April 2022.
    This can be influenced by factors, such as:
    1. Consumer purchasing power
    2 Changing consumer tastes
    3 Trends and level of market competition
    4 Certain events/holidays

    It can be said that in May 2022, when there was a 10-day Idul Fitri holiday, make-up sales decreased. 
    This is because during the Eid holiday, many business operations are also on holiday. 
    So in the case of Paragon it can be said the same way.
    During the Eid holiday, business people are faced with the challenge of staff adequacy. 
    Businesses that continue to operate during Eid, attendance arrangements, standby employee absences, and leave are crucial. 
    So that fewer employees work during the Eid holidays and sometimes the related shops also apply 2 days to one week off. 
    This causes a decrease in sales.""")
    
    # Membuat Histogram Berdasarkan Input User
    st.write('#### Histogram Based On User Input')
    pilihan = st.selectbox('Choose Column : ', ('quantity', 'week_start_date', 'week_end_date'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df1[pilihan], bins=30, kde=True)
    st.pyplot(fig)

if __name__ == '__main__':
    run()
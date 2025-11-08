import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import time


model=pk.load(open('LinearRegressionModel1.pkl','rb'))
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("""
Welcome to the **Car Price Prediction** tool!  
Fill in your car details below and get an instant price estimate 💰.
""")
st.divider()


cars_data=pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name=car_name.split(' ')[0]
    return car_name.strip()

cars_data['name']=cars_data['name'].apply(get_brand_name)

col1, col2 = st.columns(2)
with col1:
    name = st.selectbox('Car Brand', cars_data['name'].unique())
    year = st.slider('Year of Manufacture', 1994, 2025, step=1,value=2012)
    fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
    seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
    transmission = st.selectbox('Transmission', cars_data['transmission'].unique())

with col2:
    km_driven = st.number_input('Kms Driven', min_value=0, max_value=2000000, step=1000)
    owner = st.selectbox('Owner', cars_data['owner'].unique())
    mileage = st.number_input('Mileage (kmpl)', min_value=5.0, max_value=40.0, step=0.1)
    engine = st.number_input('Engine (CC)', min_value=500, max_value=5000, step=100)
    max_power = st.number_input('Max Power (bhp)', min_value=20.0, max_value=200.0, step=0.1)
    seats = st.selectbox('Seats', sorted(cars_data['seats'].dropna().unique()))


if st.button('Predict Price'):
    input_data=pd.DataFrame(
        [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']
    )


    car_price=model.predict(input_data)

    with st.spinner('Predicting...'):
        time.sleep(1.5)
        st.success(f"💰 **Predicted Car Price:** ₹ {np.round(car_price[0], 2)} Lakhs")
    
    st.markdown(f"## Predicted Car Price using Linear Regression: ₹ {np.round(car_price[0],2)} Lakhs")
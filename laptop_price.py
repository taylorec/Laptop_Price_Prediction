import streamlit as st
import pandas as pd
import numpy as np
import joblib


GBr = joblib.load('GBr.joblib')
data = pd.read_csv('laptop_data.csv')
data = data.drop(['Unnamed: 0', 'Company', 'Price'], axis=1)

st.title("Laptop Price Predictor")

# type of laptop
type_laptop = st.selectbox('Type', data['TypeName'].unique())

# Ram present in laptop
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# os of laptop
os = st.selectbox('OS', data['OpSys'].unique())

# weight of laptop
weight = st.number_input('Weight of the laptop (in lbs)')

# touchscreen available in laptop or not
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size in inches')

# resolution of laptop
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# gpu brand
gpu = st.selectbox('GPU brand', data['Gpu brand'].unique())

if st.button('Predict Price'):
    ppi = None
    weight = weight*0.454
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
        
    if type_laptop == 'Netbook':
        type_laptop = 1
    elif type_laptop == 'Notebook':
        type_laptop = 2
    elif type_laptop == 'Ultrabook':
        type_laptop = 3
    elif type_laptop == 'Gaming':
        type_laptop = 4
    elif type_laptop == '2 in 1 Convertible':
        type_laptop = 5
    elif type_laptop == 'Workstation':
        type_laptop = 6

    if os == 'Mac':
        os = 1
    elif os == 'Windows':
        os = 2
    elif os == 'Other':
        os = 3

    if cpu == 'Intel Core i3':
        cpu = 1
    elif cpu == 'Intel Core i5':
        cpu = 2
    elif cpu == 'Intel Core i7':
        cpu == 3
    elif cpu == 'Other Intel Processor':
        cpu = 4
    elif cpu == 'AMD Processor':
        cpu = 5
        
    if gpu == 'Intel':
        gpu = 1
    elif gpu == 'AMD':
        gpu = 2
    elif gpu == 'Nvidia':
        gpu = 3

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = pd.DataFrame([[type_laptop, ram, os, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu]], columns=data.columns)
    
    prediction = int(np.exp(GBr.predict(query)))

    st.title('Predicted price for this laptop: ${}' .format(prediction))

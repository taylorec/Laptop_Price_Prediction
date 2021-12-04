import streamlit as st
import pandas as pd
import numpy as np
import joblib

loaded_GBr = joblib.load('GBr.joblib')


data = pd.read_csv("laptop_data.csv")

st.title("Laptop Price Predictor")

# company name
company = st.selectbox('Brand', data['Company'].unique())

# type of laptop
type_laptop = st.selectbox('Type', data['TypeName'].unique())

# Ram present in laptop
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# os of laptop
os = st.selectbox('OS', data['OpSys'].unique())

# weight of laptop
weight = st.number_input('Weight of the laptop in kg')

# touchscreen available in laptop or not
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

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
gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = np.array([company, type_laptop, ram, os, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu])

    query = query.reshape(1, 12)

    prediction = int(np.exp(loaded_GBr.predict(query)))

    st.title('Predicted price for this laptop: ${}' .format(prediction))

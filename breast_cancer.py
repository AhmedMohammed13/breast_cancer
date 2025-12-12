import streamlit as st
import pickle
import pandas as pd


st.title('Breast Cancer Prediction App')

with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.header('Enter patient data:')

radius_mean = st.number_input('Radius Mean')
texture_mean = st.number_input('Texture Mean')
perimeter_mean = st.number_input('Perimeter Mean')
area_mean = st.number_input('Area Mean')

if st.button('Predict'):

    input_data = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean]],
                              columns=['radius_mean','texture_mean','perimeter_mean','area_mean'])
    
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        st.success('The tumor is benign')
    else:
        st.error('The tumor is malignant')

import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="♀️")

st.title("Breast Cancer Prediction App")
st.write("Enter the 4 features below to predict if the tumor is Benign or Malignant")

# تحميل الموديل
try:
    with open("breast_cancer_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    st.error("Model file not found! Make sure 'breast_cancer_model.pkl' is uploaded.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, value=13.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=85.0, step=0.1)

with col2:
    texture_mean = st.number_input("Texture Mean", min_value=0.0, value=18.0, step=0.1)
    area_mean = st.number_input("Area Mean", min_value=0.0, value=500.0, step=1.0)

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean]
    })
    
    prediction = model.predict(input_df)[0]
    
    if prediction == 0:
        st.success("The tumor is **Benign** (Non-cancerous)")
        st.balloons()
    else:
        st.error("The tumor is **Malignant** (Cancerous)")
        st.warning("Please consult a doctor immediately") 

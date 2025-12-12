import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Breast Cancer Prediction")

st.title("Breast Cancer Prediction App")
st.write("Enter the four measurements to predict if the tumor is Benign or Malignant.")

try:
    with open("breast_cancer_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'breast_cancer_model.pkl' not found in the repository!")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.header("Enter Patient Data")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, value=13.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=85.0, step=0.1)

with col2:
    texture_mean = st.number_input("Texture Mean", min_value=0.0, value=18.0, step=0.1)
    area_mean = st.number_input("Area Mean", min_value=0.0, value=500.0, step=1.0)

if st.button("Predict", type="primary"):
    # إنشاء الإنبوت بنفس ترتيب وأسماء الفيتشرات اللي الموديل مدرب عليها
    input_data = pd.DataFrame([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean
    ]], columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'])

    try:
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.success(" The tumor is **Benign** (Non-cancerous)")
            st.balloons()
        else:
            st.error(" The tumor is **Malignant** (Cancerous)")
            st.warning("This is a prediction only. Please consult a medical professional.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

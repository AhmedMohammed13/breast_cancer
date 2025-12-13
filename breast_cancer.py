import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(page_title="Breast Cancer Prediction")

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the four key measurements below to predict whether the tumor is **Benign** (non-cancerous) or **Malignant** (cancerous).")
st.write("**Important Note:** This is a machine learning prediction only — always consult a medical professional for an accurate diagnosis.")

try:
    with open("breast_cancer_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        scaler = artifacts['scaler']
except FileNotFoundError:
    st.error("Model file 'breast_cancer_model.pkl' not found in the repository!")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.header("Enter Patient Data")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input(
        "Radius Mean",
        min_value=6.0, max_value=30.0, value=13.0, step=0.1,
        help="Average radius of tumor cell nuclei (typical range: 6–30)"
    )
    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=40.0, max_value=190.0, value=85.0, step=0.1,
        help="Average perimeter of tumor cell nuclei (typical range: 40–190)"
    )

with col2:
    texture_mean = st.number_input(
        "Texture Mean",
        min_value=9.0, max_value=40.0, value=18.0, step=0.1,
        help="Average texture (standard deviation of gray-scale values) (typical range: 9–40)"
    )
    area_mean = st.number_input(
        "Area Mean",
        min_value=140.0, max_value=2500.0, value=550.0, step=1.0,
        help="Average area of tumor cell nuclei (typical range: 140–2500)"
    )

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame({
        'mean radius': [radius_mean],
        'mean texture': [texture_mean],
        'mean perimeter': [perimeter_mean],
        'mean area': [area_mean]
    })

    try:
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        if prob[1] > prob[0]:
            st.success(" The tumor is **Benign** (Non-cancerous)")
            st.balloons()
        else:
            st.error(" The tumor is **Malignant** (Cancerous)")
            st.warning("This is a prediction only. Please consult a medical professional immediately.")

        st.write(f"**Probability of Benign**: {prob[1]:.2%}")
        st.write(f"**Probability of Malignant**: {prob[0]:.2%}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

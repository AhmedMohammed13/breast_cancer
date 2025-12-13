import streamlit as st
import pickle
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Breast Cancer Prediction")

# العنوان والوصف
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the four key measurements below to predict whether the tumor is **Benign** (non-cancerous) or **Malignant** (cancerous).")
st.write("**Note:** This is a machine learning prediction only — always consult a medical professional for diagnosis.")

# تحميل الموديل
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
    radius_mean = st.number_input("Radius Mean", min_value=6.0, max_value=30.0, value=13.0, step=0.1,
                                  help="Average radius of the tumor cell nuclei (typical range: 6–30)")
    perimeter_mean = st.number_input("Perimeter Mean", min_value=40.0, max_value=190.0, value=85.0, step=0.1,
                                     help="Average perimeter of the tumor cell nuclei (typical range: 40–190)")

with col2:
    texture_mean = st.number_input("Texture Mean", min_value=9.0, max_value=40.0, value=18.0, step=0.1,
                                   help="Average texture (standard deviation of gray-scale values) (typical range: 9–40)")
    area_mean = st.number_input("Area Mean", min_value=140.0, max_value=2500.0, value=550.0, step=1.0,
                                help="Average area of the tumor cell nuclei (typical range: 140–2500)")

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean]
    })

    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

     
        if prediction == 1:
            st.success(" The tumor is **Benign** (Non-cancerous)")
            st.balloons()
        else:  # prediction == 0
            st.error(" The tumor is **Malignant** (Cancerous)")
            st.warning("This is a prediction only. Please consult a medical professional immediately.")

        st.write(f"**Probability of Benign**: {prob[1]:.2%}")
        st.write(f"**Probability of Malignant**: {prob[0]:.2%}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Built with  using Streamlit | Model: Random Forest on Breast Cancer Wisconsin Dataset")

import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Load your trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Main title
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts whether a person is **diabetic** or **not diabetic** based on 8 health inputs.

Fill out the form below to get your result!
""")

# Sidebar for input values
st.sidebar.header("Enter Patient Information")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)

# Prediction function
def predict_diabetes(data):
    data_np = np.array(data).reshape(1, -1)
    prediction = model.predict(data_np)
    return prediction[0]

# Main form
# ... previous imports and model loading remain unchanged ...

# Prediction result
if st.button("Predict Diabetes Status"):
    user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    result = predict_diabetes(user_input)
    
    if result == 1:
        st.error("🚨 Based on the data, there's a possibility of diabetes.\n\nPlease consult a healthcare professional for further evaluation and guidance.")
    else:
        st.success("🎉 The prediction suggests no signs of diabetes.\n\nKeep maintaining a healthy lifestyle and regular checkups!")


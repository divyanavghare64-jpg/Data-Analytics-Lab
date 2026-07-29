
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# App Title and Description
st.title("🩺 Diabetes Prediction App (Binary Classification)")
st.write("Enter the required medical details below to check the diabetes prediction result.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction Button
if st.button("Predict Result"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("🚨 Result: The model predicts a high probability of Diabetes.")
    else:
        st.success("✅ Result: The model predicts No Diabetes. You look safe!")
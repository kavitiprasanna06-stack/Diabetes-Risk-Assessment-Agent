import streamlit as st
import pickle
import numpy as np

# Load Model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Scaler
with open("daibetes_model.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")

st.write("Enter the patient details below.")

preg = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
skin = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.50)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    data = np.array([[preg,
                      glucose,
                      bp,
                      skin,
                      insulin,
                      bmi,
                      dpf,
                      age]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is not likely to have Diabetes.")

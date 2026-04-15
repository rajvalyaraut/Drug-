import streamlit as st
import pickle
import numpy as np

# Load model
with open("Drug Prediction Model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="ML Prediction App")

st.title("🚀 Machine Learning Deployment App")

st.write("Enter input values below:")

# Example inputs (modify based on your dataset)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# Prediction button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")

import streamlit as st
import requests

st.title("Creditwise AI Dashboard")

age = st.number_input("Age", min_value=18, max_value=60)
income = st.number_input("Income", 50000, 500000, step=1000)
defaults = st.number_input("Previous Defaults", 0, 10)
requested_amount = st.number_input("Requested Loan Amount", 1000, 50000, step=500)

if st.button("Predict Loan Approval"):
    payload = {
        "age": age,
        "income": income,
        "requested_amount": requested_amount,
        "previous_defaults": defaults
    }
    response = requests.post("http://localhost:8000/predict", json=payload).json()
    st.write("Approval:", response["approval"])
    st.write("Confidence:", response["confidence"])
    st.write("Suggested Amount:", response["suggested_amount"])
    
    shap_resp = requests.post("http://localhost:8000/explain", json=payload).json()
    st.write("Classification Explanation:", shap_resp["classification_explanation"])
    st.write("Regression Explanation:", shap_resp["regression_explanation"])
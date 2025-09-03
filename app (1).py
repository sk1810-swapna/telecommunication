# -*- coding: utf-8 -*-
"""app.py — Telecom Churn Risk Tiering"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title and description
st.title("📉 Telecom Churn Risk Predictor")
st.markdown("""
Telecom companies face annual churn rates over 10%.  
This app estimates the **probability of churn** for individual customers and classifies them into **risk tiers** to guide retention strategies.
""")

# Load model and scaler
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("❌ Missing model or scaler file. Please upload 'rf_model.pkl' and 'scaler.pkl'.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Module error: {e}. Check your environment setup.")
    st.stop()

# Sidebar inputs
st.sidebar.header("📋 Customer Features")
def get_input():
    account_length = st.sidebar.number_input("Account Length", min_value=1, max_value=300, value=100)
    customer_service_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
    day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, max_value=500.0, value=120.0)
    total_charge = st.sidebar.number_input("Total Charge", min_value=0.0, max_value=100.0, value=45.0)

    return pd.DataFrame([{
        'account_length': account_length,
        'customer_service_calls': customer_service_calls,
        'day_mins': day_mins,
        'total_charge': total_charge
    }])

input_df = get_input()

# Show input
st.subheader("🔍 Customer Profile")
st.write(input_df)

# Risk tier logic
def classify_risk(prob):
    if prob < 0.30:
        return "🟢 Low Risk", "Customer is unlikely to churn. Maintain satisfaction."
    elif prob < 0.70:
        return "🟡 Medium Risk", "Customer shows moderate churn risk. Monitor engagement."
    else:
        return "🔴 High Risk", "Customer is likely to churn. Consider proactive retention."

# Predict button
if st.button("📊 Estimate Churn Risk"):
    try:
        # Align input with scaler
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[expected_features]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Classify risk
        risk_label, advice = classify_risk(prediction_proba)

        # Display result
        st.subheader("🧠 Prediction Result")
        st.write(f"**Estimated Churn Probability:** `{prediction_proba:.2%}`")
        st.markdown(f"**Risk Tier:** {risk_label}")
        st.info(advice)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

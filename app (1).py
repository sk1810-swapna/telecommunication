# -*- coding: utf-8 -*-
"""app.py — Telecom Churn Probability Modeling"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title and project description
st.title("📉 Telecom Churn Probability Model")
st.markdown("""
Customer churn is a major challenge for telecom companies, with annual churn rates often exceeding 10%.  
This app models the probability of churn based on customer features, helping identify clients at risk and guiding retention strategies.
""")

# Load model and scaler
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("❌ Required files not found. Please ensure 'rf_model.pkl' and 'scaler.pkl' are present.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Module error: {e}. Check your environment setup.")
    st.stop()

# Input form
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

# Display input
st.subheader("🔍 Customer Profile")
st.write(input_df)

# Prediction button
if st.button("📊 Predict Churn Probability"):
    try:
        # Align input with scaler
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[expected_features]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Output
        st.subheader("🧠 Prediction Result")
        st.write(f"**Estimated Churn Probability:** `{prediction_proba:.2%}`")

        if prediction[0] == 1:
            st.error("⚠️ This customer is likely to churn. Consider proactive retention strategies.")
        else:
            st.success("✅ This customer is likely to stay. Maintain engagement and satisfaction.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

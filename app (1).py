# -*- coding: utf-8 -*-
"""app.py — Binary Churn Prediction"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title and description
st.title("📉 Telecom Churn Classifier")
st.markdown("""
This app predicts whether a telecom customer is likely to **churn (1)** or **stay loyal (0)**  
based on key behavioral and usage features.  
Use it to identify high-risk customers and guide retention strategies.
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

# Sidebar inputs using sliders
st.sidebar.header("📋 Customer Features")
def get_input():
    account_length = st.sidebar.slider("Account Length", min_value=1, max_value=300, value=100)
    customer_service_calls = st.sidebar.slider("Customer Service Calls", min_value=0, max_value=20, value=1)
    day_mins = st.sidebar.slider("Day Minutes", min_value=0.0, max_value=500.0, value=120.0)
    total_charge = st.sidebar.slider("Total Charge", min_value=0.0, max_value=100.0, value=45.0)

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

# Predict button
if st.button("📊 Predict Churn (Binary)"):
    try:
        # Align input with scaler
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[expected_features]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Display result
        st.subheader("🧠 Prediction Result")
        st.write(f"**Binary Churn Prediction:** `{prediction}`")

        if prediction == 1:
            st.error("⚠️ This customer is predicted to churn.")
        else:
            st.success("✅ This customer is predicted to stay loyal.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

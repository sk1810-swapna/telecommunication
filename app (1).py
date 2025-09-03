# -*- coding: utf-8 -*-
"""app.py — Telecom Churn Prediction"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# App title
st.title("📊 Telecom Churn Prediction")
st.markdown("Enter customer details to predict churn likelihood.")

# Load model and scaler safely
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please upload 'rf_model.pkl' and 'scaler.pkl'.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module during model loading: {e}. Make sure all dependencies are installed.")
    st.stop()

# Sidebar input fields
def user_input():
    account_length = st.number_input("Account Length", min_value=1, max_value=300, value=100)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
    day_mins = st.number_input("Day Minutes", min_value=0.0, max_value=500.0, value=120.0)
    total_charge = st.number_input("Total Charge", min_value=0.0, max_value=100.0, value=45.0)

    input_dict = {
        'account_length': account_length,
        'customer_service_calls': customer_service_calls,
        'day_mins': day_mins,
        'total_charge': total_charge
    }
    return pd.DataFrame([input_dict])

input_df = user_input()

# Display input
st.subheader("Customer Input")
st.write(input_df)

# Add prediction button
if st.button("🔍 Predict Churn"):
    try:
        # Align input features with scaler expectations
        expected_features = scaler.feature_names_in_
        missing_cols = [col for col in expected_features if col not in input_df.columns]

        # Fill missing columns with default values
        for col in missing_cols:
            input_df[col] = 0.0

        # Reorder columns to match scaler
        input_df = input_df[expected_features]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Show result
        st.subheader("Prediction Result")
        

        # Show probability
        st.write(f"Probability of churn: {prediction_proba[0][1]:.2%}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature list
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_features = scaler.feature_names_in_.tolist()

# Title
st.title("📉 Churn Prediction App")

# Sidebar inputs
st.sidebar.header("Customer Info")
account_length = st.sidebar.slider("Account Length", 1, 250, 100)
customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)
total_charge = st.sidebar.slider("Total Charge ($)", 0.0, 200.0, 75.0)

day_mins = st.sidebar.slider("Day Minutes", 0.0, 350.0, 180.0)
day_calls = st.sidebar.slider("Day Calls", 0, 200, 100)
day_charge = st.sidebar.slider("Day Charge", 0.0, 60.0, 30.0)

evening_mins = st.sidebar.slider("Evening Minutes", 0.0, 350.0, 180.0)
evening_calls = st.sidebar.slider("Evening Calls", 0, 200, 100)
evening_charge = st.sidebar.slider("Evening Charge", 0.0, 60.0, 30.0)

night_mins = st.sidebar.slider("Night Minutes", 0.0, 350.0, 180.0)
night_calls = st.sidebar.slider("Night Calls", 0, 200, 100)
night_charge = st.sidebar.slider("Night Charge", 0.0, 60.0, 30.0)

intl_mins = st.sidebar.slider("International Minutes", 0.0, 20.0, 10.0)
intl_calls = st.sidebar.slider("International Calls", 0, 20, 5)
intl_charge = st.sidebar.slider("International Charge", 0.0, 5.0, 2.5)

# Prepare input
input_data = pd.DataFrame([[
    account_length, customer_service_calls, total_charge,
    day_calls, day_charge, day_mins,
    evening_calls, evening_charge, evening_mins,
    night_calls, night_charge, night_mins,
    intl_calls, intl_charge, intl_mins
]], columns=expected_features)

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input)[0]
proba = model.predict_proba(scaled_input)[0][1]

# Output
if prediction == 1:
    st.success(f"⚠️ This customer is likely to churn. (Probability: {proba:.2f})")
else:
    st.info(f"✅ This customer is likely to stay. (Probability of churn: {proba:.2f})")

# Diagnostic check
try:
    test_data = pd.DataFrame(np.random.rand(100, len(expected_features)) * 100, columns=expected_features)
    test_scaled = scaler.transform(test_data)
    unique_preds = np.unique(model.predict(test_scaled))
    if len(unique_preds) == 1:
        st.warning("⚠️ Model is predicting only one class. Consider retraining with more balanced data.")
except Exception as e:
    st.error(f"Diagnostic check failed: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define expected feature order
expected_features = ['account_length', 'customer_service_calls', 'total_charge']

# Title
st.title("📉 Churn Prediction App")

# Sidebar inputs
st.sidebar.header("Customer Info")
account_length = st.sidebar.slider("Account Length", 1, 250, 100)
customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)
total_charge = st.sidebar.slider("Total Charge ($)", 0.0, 200.0, 75.0)

# Prepare input with correct feature names and order
input_data = pd.DataFrame([[account_length, customer_service_calls, total_charge]], columns=expected_features)

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
    test_data = pd.DataFrame({
        'account_length': np.random.randint(1, 250, 100),
        'customer_service_calls': np.random.randint(0, 10, 100),
        'total_charge': np.random.uniform(0, 200, 100)
    })[expected_features]
    unique_preds = np.unique(model.predict(scaler.transform(test_data)))
    if len(unique_preds) == 1:
        st.warning("⚠️ Model is predicting only one class. Consider retraining with more balanced data.")
except Exception as e:
    st.error(f"Diagnostic check failed: {e}")

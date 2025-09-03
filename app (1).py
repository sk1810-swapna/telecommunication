# -*- coding: utf-8 -*-
"""app.py — Binary Churn Prediction"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🎯 Title and description
st.title("📉 Telecom Churn Classifier")
st.markdown("""
This app predicts whether a telecom customer is likely to **churn (1)** or **stay loyal (0)**  
based on key behavioral and usage features.  
Use it to identify high-risk customers and guide retention strategies.
""")

# 📦 Load model and scaler
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("❌ Missing model or scaler file. Please upload 'rf_model.pkl' and 'scaler.pkl'.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Module error: {e}. Check your environment setup.")
    st.stop()

# 📋 Sidebar inputs
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

# 🔍 Show input
st.subheader("🔍 Customer Profile")
st.write(input_df)

# 📊 Predict churn
if st.button("📊 Predict Churn (Binary)"):
    try:
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[expected_features]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.subheader("🧠 Prediction Result")
        st.write(f"**Binary Churn Prediction:** `{prediction}`")

        if prediction == 1:
            st.error("⚠️ This customer is predicted to churn.")
        else:
            st.success("✅ This customer is predicted to stay loyal.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# 📈 Feature Importance
st.subheader("📌 Feature Importance")
try:
    importances = model.feature_importances_
    feature_names = scaler.feature_names_in_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)
except Exception as e:
    st.warning(f"⚠️ Could not display feature importance: {e}")

# 🔁 Sensitivity Analysis
st.subheader("🔄 Sensitivity: Vary Account Length")
try:
    sample = input_df.copy()
    expected_features = scaler.feature_names_in_
    for col in expected_features:
        if col not in sample.columns:
            sample[col] = 0.0
    sample = sample[expected_features]

    results = []
    flip_point = None
    for val in range(50, 201, 10):
        sample['account_length'] = val
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)[0]
        results.append((val, pred))
        if pred == 0 and flip_point is None:
            flip_point = val

    sensitivity_df = pd.DataFrame(results, columns=['Account Length', 'Churn Prediction'])
    st.line_chart(sensitivity_df.set_index('Account Length'))
    st.dataframe(sensitivity_df)

    if flip_point:
        st.success(f"✅ Prediction flips to 'Stay Loyal' when Account Length reaches **{flip_point}**.")
    else:
        st.warning("⚠️ Prediction did not flip to 'Stay Loyal' within tested range.")

except Exception as e:
    st.warning(f"⚠️ Sensitivity analysis failed: {e}")


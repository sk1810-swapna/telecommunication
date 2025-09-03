import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Get expected feature names from scaler
expected_features = scaler.feature_names_in_.tolist()

# Title
st.title("📉 Churn Prediction App")

# Sidebar inputs
st.sidebar.header("Customer Info")

# Create input dictionary dynamically
input_dict = {}
for feature in expected_features:
    label = feature.replace('_', ' ').title()
    if "length" in feature or "calls" in feature or "messages" in feature:
        input_dict[feature] = st.sidebar.slider(label, 0, 250, 50)
    elif "mins" in feature:
        input_dict[feature] = st.sidebar.slider(label, 0.0, 350.0, 180.0)
    elif "charge" in feature:
        input_dict[feature] = st.sidebar.slider(label, 0.0, 60.0, 30.0)
    else:
        input_dict[feature] = st.sidebar.slider(label, 0.0, 100.0, 50.0)

# Align input with expected features
try:
    input_data = pd.DataFrame([input_dict])[expected_features]
    scaled_input = scaler.transform(input_data)
except Exception as e:
    st.error(f"❌ Input preparation or scaling failed: {e}")
    st.stop()

# Predict
try:
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# Output in binary format
# Output
if prediction == 1:
    st.success(f"⚠️ This customer is likely to churn. (Probability: {proba:.2f})")
else:
    st.info(f"✅ This customer is likely to stay. (Probability of churn: {proba:.2f})")
st.subheader("🔢 Churn Prediction (Binary Output)")
st.code(f"{prediction}", language="text")

# Optional: show probability
with st.expander("Show Prediction Probability"):
    st.write(f"Churn Probability: {proba:.2f}")

# Diagnostic check for class diversity
try:
    test_data = pd.DataFrame(np.random.rand(100, len(expected_features)) * 100, columns=expected_features)
    test_scaled = scaler.transform(test_data)
    unique_preds = np.unique(model.predict(test_scaled))
    if len(unique_preds) == 1:
        st.warning("⚠️ Model is predicting only one class. Consider retraining with more balanced data.")
except Exception as e:
    st.error(f"Diagnostic check failed: {e}")

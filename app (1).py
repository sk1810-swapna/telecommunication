# telecom_churn_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")

# --- Load or Train Model ---
def load_or_train_model():
    if os.path.exists("rf_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
    else:
        df = pd.read_csv("telecom_churn.csv")

        # Basic cleaning
        df.dropna(inplace=True)
        df = df[df['churn'].isin([0, 1])]

        X = df.drop(columns=['churn'])
        y = df['churn']

        # Balance data
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        # Train model
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(X_scaled, y_resampled)

        # Save model
        joblib.dump(model, "rf_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

    return model, scaler

model, scaler = load_or_train_model()

# --- User Input ---
st.title("📞 Telecom Churn Predictor")
st.markdown("Enter customer details to predict churn:")

def user_input_features():
    account_length = st.slider("Account Length", 1, 250, 100)
    customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)
    international_plan = st.selectbox("International Plan", ["Yes", "No"])
    voice_mail_plan = st.selectbox("Voice Mail Plan", ["Yes", "No"])
    total_day_minutes = st.slider("Total Day Minutes", 0.0, 400.0, 180.0)
    total_eve_minutes = st.slider("Total Evening Minutes", 0.0, 400.0, 180.0)
    total_night_minutes = st.slider("Total Night Minutes", 0.0, 400.0, 180.0)
    total_intl_minutes = st.slider("Total Intl Minutes", 0.0, 20.0, 10.0)

    data = {
        "account_length": account_length,
        "customer_service_calls": customer_service_calls,
        "international_plan": 1 if international_plan == "Yes" else 0,
        "voice_mail_plan": 1 if voice_mail_plan == "Yes" else 0,
        "total_day_minutes": total_day_minutes,
        "total_eve_minutes": total_eve_minutes,
        "total_night_minutes": total_night_minutes,
        "total_intl_minutes": total_intl_minutes
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# --- Prediction ---
if st.button("Predict Churn"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. Confidence: {prediction_proba:.2f}")
    else:
        st.success(f"✅ This customer is likely to stay. Confidence: {prediction_proba:.2f}")

# --- Diagnostic ---
def check_model_bias(model, X):
    preds = model.predict(X)
    unique_preds = np.unique(preds)
    if len(unique_preds) == 1:
        st.warning("⚠️ Model is predicting only one class. Consider retraining with more balanced data.")

# Run diagnostic
X_sample = scaler.transform(input_df)
check_model_bias(model, X_sample)

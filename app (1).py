# telecom_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Page Config ---
st.set_page_config(page_title="📞 Telecom Churn Predictor", layout="centered")

# --- Load or Train Model ---
def load_or_train_model():
    if all(os.path.exists(f) for f in ["rf_model.pkl", "scaler.pkl", "feature_names.pkl"]):
        model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
    else:
        try:
            df = pd.read_csv("telecommunications_churn (1).csv")
        except FileNotFoundError:
            st.error("❌ Dataset 'telecommunications_churn (1).csv' not found. Please upload it to the app directory.")
            st.stop()

        df.drop(columns=['international_plan', 'voice_mail_plan'], errors='ignore', inplace=True)
        df.dropna(inplace=True)
        df = df[df['churn'].isin([0, 1])]

        X = df.drop(columns=['churn'])
        y = df['churn']

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(X_scaled, y_resampled)

        unique_preds = np.unique(model.predict(X_scaled))
        if len(unique_preds) == 1:
            st.error(f"❌ Model is predicting only one class ({unique_preds[0]}) after training. Please check your data balance.")
            st.stop()

        feature_names = X.columns.tolist()
        joblib.dump(model, "rf_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(feature_names, "feature_names.pkl")

    return model, scaler, feature_names

model, scaler, feature_names = load_or_train_model()

# --- Title ---
st.title("📞 Telecom Churn Predictor")
st.markdown("Use the sidebar to enter customer details. The app will predict whether the customer is likely to churn.")

# --- Sidebar Input ---
st.sidebar.header("📋 Customer Details")
def user_input_features():
    account_length = st.sidebar.slider("Account Length", 1, 250, 100)
    customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)
    total_day_minutes = st.sidebar.slider("Total Day Minutes", 0.0, 400.0, 180.0)
    total_eve_minutes = st.sidebar.slider("Total Evening Minutes", 0.0, 400.0, 180.0)
    total_night_minutes = st.sidebar.slider("Total Night Minutes", 0.0, 400.0, 180.0)
    total_intl_minutes = st.sidebar.slider("Total Intl Minutes", 0.0, 20.0, 10.0)

    data = {
        "account_length": account_length,
        "customer_service_calls": customer_service_calls,
        "total_day_minutes": total_day_minutes,
        "total_eve_minutes": total_eve_minutes,
        "total_night_minutes": total_night_minutes,
        "total_intl_minutes": total_intl_minutes
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# --- Align Input with Training Features ---
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]
input_df.columns.name = None
input_df = input_df.astype(float)

# --- Display Summary Table ---
st.subheader("📊 Input Summary")
st.dataframe(input_df.style.format(precision=2), use_container_width=True)

# --- Prediction ---
if st.button("🔍 Predict Churn"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][prediction]

        # Display binary prediction
        st.subheader("🔢 Churn Prediction")
        st.code(f"{prediction}", language="text")

        # Add a short message based on prediction
        amsg = "Customer is likely to churn." if prediction == 1 else "Customer is likely to stay."
        st.write(f"🗨️ {amsg}")

        # Optional: show confidence score
        with st.expander("Show Prediction Confidence"):
            st.write(f"Confidence: {prediction_proba:.2f}")

    except ValueError as e:
        st.error(f"❌ Prediction failed due to input mismatch: {e}")

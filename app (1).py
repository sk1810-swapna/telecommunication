import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Page Config ---
st.set_page_config(page_title="📞 Telecom Churn Predictor", layout="centered")
st.title("📞 Telecom Churn Predictor")
st.markdown("Use the sidebar to enter customer details and choose a model. The app will predict whether the customer is likely to churn.")

# --- Sidebar Inputs ---
st.sidebar.header("📋 Customer Details")
algorithm = st.sidebar.selectbox("Choose Classification Algorithm", [
    "Random Forest", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"
])

raw_inputs = {
    "Account Length": st.sidebar.slider("Account Length", 1, 250, 100),
    "Customer Service Calls": st.sidebar.slider("Customer Service Calls", 0, 10, 1),
    "Total Day Minutes": st.sidebar.slider("Total Day Minutes", 0.0, 400.0, 180.0),
    "Total Evening Minutes": st.sidebar.slider("Total Evening Minutes", 0.0, 400.0, 180.0),
    "Total Night Minutes": st.sidebar.slider("Total Night Minutes", 0.0, 400.0, 180.0),
    "Total Intl Minutes": st.sidebar.slider("Total Intl Minutes", 0.0, 20.0, 10.0)
}

# --- Load and Preprocess Dataset ---
try:
    df = pd.read_csv("telecommunications_churn (1).csv")
except FileNotFoundError:
    st.error("❌ Dataset 'telecommunications_churn (1).csv' not found.")
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

feature_names = X.columns.tolist()

# --- Train Model Based on Selection ---
if algorithm == "Random Forest":
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
elif algorithm == "Support Vector Machine":
    model = SVC(class_weight='balanced', probability=True)
elif algorithm == "Decision Tree":
    model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
elif algorithm == "K-Nearest Neighbors":
    model = KNeighborsClassifier()

model.fit(X_scaled, y_resampled)

# --- Prepare Input for Prediction ---
summary_df = pd.DataFrame([raw_inputs])
st.subheader("📊 Input Summary")
st.dataframe(summary_df.style.format(precision=2), use_container_width=True)

model_input = {
    "account_length": raw_inputs["Account Length"],
    "customer_service_calls": raw_inputs["Customer Service Calls"],
    "total_day_minutes": raw_inputs["Total Day Minutes"],
    "total_eve_minutes": raw_inputs["Total Evening Minutes"],
    "total_night_minutes": raw_inputs["Total Night Minutes"],
    "total_intl_minutes": raw_inputs["Total Intl Minutes"]
}

input_df = pd.DataFrame([model_input])
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names].astype(float)
input_scaled = scaler.transform(input_df)

# --- Predict ---
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]

    st.subheader("🔢 Churn Prediction")
    st.code(f"{prediction}", language="text")

    amsg = "Customer is likely to churn." if prediction == 1 else "Customer is likely to stay."
    st.write(f"🗨️ {amsg}")

    with st.expander("Show Prediction Confidence"):
        st.write(f"Confidence: {prediction_proba:.2f}")


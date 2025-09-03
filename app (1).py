import streamlit as st
import pandas as pd
import cloudpickle

# Load model and scaler using cloudpickle
def load_pickle(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)

model = load_pickle("model.pkl")
scaler = load_pickle("scaler.pkl")

st.title("📞 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn based on their service usage.")

# Input fields
customer_service_calls = st.number_input("Number of Customer Service Calls", min_value=0, step=1)
account_length = st.number_input("Account Length", min_value=0, step=1)
international_plan = st.selectbox("International Plan", ["Yes", "No"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["Yes", "No"])
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, step=0.1)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, step=0.1)
total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, step=0.1)
total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, step=0.1)

# Prepare input DataFrame
input_dict = {
    "account_length": account_length,
    "international_plan": 1 if international_plan == "Yes" else 0,
    "voice_mail_plan": 1 if voice_mail_plan == "Yes" else 0,
    "number_customer_service_calls": customer_service_calls,
    "total_day_minutes": total_day_minutes,
    "total_eve_minutes": total_eve_minutes,
    "total_night_minutes": total_night_minutes,
    "total_intl_minutes": total_intl_minutes
}

input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("📊 Predict Churn (Binary)"):
    try:
        # Filter: Only predict if service calls > 10
        if input_df['number_customer_service_calls'].iloc[0] <= 10:
            st.warning("⚠️ Prediction skipped: Customer must have made more than 10 service calls.")
        else:
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

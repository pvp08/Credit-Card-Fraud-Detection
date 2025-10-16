import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    "### Enter transaction details below and select a trained model to predict whether the transaction is **fraudulent or legitimate.**"
)

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Random Forest", "Logistic Regression", "XGBoost")
)

# Load model dynamically
model_path = f"../models/{model_choice.lower().replace(' ', '_')}.pkl"
model = joblib.load(model_path)

st.sidebar.success(f"Loaded: {model_choice}")

# ---- INPUT SECTION ----
st.markdown("### Transaction Features")

col1, col2 = st.columns(2)

# Left column inputs
with col1:
    time = st.slider("⏱ Transaction Time (sec since first)", 0, 200000, 100000)
    amount = st.slider("💰 Transaction Amount ($)", 0.0, 2500.0, 100.0)

# Right column inputs (V1–V28)
with col2:
    st.caption("Enter anonymized PCA components (V1–V28)")
    v_features = [st.number_input(f"V{i}", value=0.0, step=0.001, format="%.3f") for i in range(1, 29)]

# Combine all features
input_data = np.array([time] + v_features + [amount]).reshape(1, -1)

# ---- PREDICTION ----
st.markdown("---")
if st.button("🔍 Predict Transaction Type"):
    proba = model.predict_proba(input_data)[0][1]  # Probability of fraud
    prediction = model.predict(input_data)[0]

    st.markdown("### 🧠 Model Prediction:")
    if prediction == 0:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2%})")
    else:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Fraud Probability: {proba:.2%})")

    # ---- GAUGE VISUALIZATION ----
    st.markdown("### 🔎 Fraud Probability Gauge")
    st.progress(int(proba * 100))
    st.caption(f"{proba:.2%} likelihood of fraud detected by {model_choice}")

    # ---- Display Feature Summary ----
    feature_df = pd.DataFrame(input_data, columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    st.markdown("### 📋 Input Summary")
    st.dataframe(feature_df)

st.markdown("---")
st.markdown("🧠 *This app uses machine learning models trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).*")
st.caption("© 2025 Credit Card Fraud Detection App | Built with ❤️ using Streamlit")

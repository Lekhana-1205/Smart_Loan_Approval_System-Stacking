import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🎯",
    layout="wide"
)
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

scaler = load_scaler()

# ---------------------------------
# Load Model & Metadata Safely
# ---------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("stacking_loan_model.pkl"):
        st.error("❌ Model file not found. Please run train_model.py first.")
        st.stop()

    if not os.path.exists("model_features.pkl"):
        st.error("❌ Feature file not found. Please run train_model.py first.")
        st.stop()

    classifier = joblib.load("stacking_loan_model.pkl")
    features = joblib.load("model_features.pkl")

    return classifier, features


classifier, features = load_artifacts()

# ---------------------------------
# Title & Description
# ---------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")

st.markdown(
    """
    This system uses a **Stacking Ensemble Machine Learning model** to predict  
    whether a loan will be approved by combining multiple ML models  
    for better and more stable decision-making.
    """
)

st.divider()

# ---------------------------------
# Sidebar Inputs
# ---------------------------------
st.sidebar.header("🧾 Applicant Details")

ApplicantIncome = st.sidebar.number_input(
    "Applicant Income", min_value=0, step=1000
)
CoapplicantIncome = st.sidebar.number_input(
    "Co-Applicant Income", min_value=0, step=1000
)
LoanAmount = st.sidebar.number_input(
    "Loan Amount", min_value=0, step=100
)
Loan_Amount_Term = st.sidebar.number_input(
    "Loan Amount Term (Months)", min_value=0, step=12
)

Credit_History = st.sidebar.radio(
    "Credit History", ["Yes", "No"]
)
Employment_Status = st.sidebar.selectbox(
    "Employment Status", ["Salaried", "Self-Employed"]
)
Property_Area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semi-Urban", "Rural"]
)

# ---------------------------------
# Encode Inputs (MATCH TRAINING)
# ---------------------------------
input_data = {
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": 1 if Credit_History == "Yes" else 0,
    "Employment_Status": 1 if Employment_Status == "Salaried" else 0,
    "Property_Area": {
        "Rural": 0,
        "Semi-Urban": 1,
        "Urban": 2
    }[Property_Area],
}

input_df = pd.DataFrame([input_data])

# Ensure correct column order
input_df = input_df.reindex(columns=features, fill_value=0)

# ---------------------------------
# Model Architecture Display
# ---------------------------------
st.subheader("🧩 Stacking Model Architecture")

st.markdown(
    """
    **Base Models Used**
    - Logistic Regression  
    - Decision Tree  
    - Random Forest  

    **Meta Model**
    - Logistic Regression  

    The final prediction is made by combining  
    the outputs of all base models.
    """
)

st.divider()

# ---------------------------------
# Prediction Button
# ---------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Base model predictions
    lr_pred = classifier.named_estimators_["lr"].predict(input_df)[0]
    dt_pred = classifier.named_estimators_["dt"].predict(input_df)[0]
    rf_pred = classifier.named_estimators_["rf"].predict(input_df)[0]

    # Final stacking prediction
    input_scaled = scaler.transform(input_df)

    final_pred = classifier.predict(input_scaled)[0]
    confidence = classifier.predict_proba(input_scaled)[0][final_pred]

    # ---------------------------------
    # Output Section
    # ---------------------------------
    st.subheader("📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write("Approved" if final_pred else "Rejected")

    st.markdown(f"### 📈 Confidence Score: {confidence:.2f}%")

    # ---------------------------------
    # Business Explanation (MANDATORY)
    # ---------------------------------
    st.divider()
    st.subheader("💼 Business Explanation")

    if final_pred == 1:
        st.info(
            "Based on the applicant’s income, credit history, and the combined "
            "predictions from multiple machine learning models, the applicant is "
            "considered likely to repay the loan. Therefore, the stacking model "
            "predicts loan approval."
        )
    else:
        st.info(
            "Based on the applicant’s income, credit history, and the combined "
            "predictions from multiple machine learning models, the applicant is "
            "considered unlikely to repay the loan. Therefore, the stacking model "
            "predicts loan rejection."
        )

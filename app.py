import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Smart Loan Approval System", page_icon="🎯", layout="wide")

# ---------------- Load Model Files ----------------
@st.cache_resource
def load_files():
    model = joblib.load("stacking_loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

classifier, scaler, features = load_files()

# ---------------- Title ----------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write("This system uses a **Stacking Ensemble ML Model** to predict loan approval.")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🧾 Applicant Details")

ApplicantIncome = st.sidebar.number_input("Applicant Income", 0)
CoapplicantIncome = st.sidebar.number_input("Co-Applicant Income", 0)
LoanAmount = st.sidebar.number_input("Loan Amount", 0)
Loan_Amount_Term = st.sidebar.number_input("Loan Term (Months)", 0)

Credit_History = st.sidebar.radio("Credit History", ["Yes", "No"])
Employment_Status = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# ---------------- Encode Inputs ----------------
input_data = {
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": 1 if Credit_History == "Yes" else 0,
    "Employment_Status": 1 if Employment_Status == "Salaried" else 0,
    "Property_Area": {"Rural": 0, "Semi-Urban": 1, "Urban": 2}[Property_Area]
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=features, fill_value=0)
input_scaled = scaler.transform(input_df)

# ---------------- Model Architecture Section ----------------
st.subheader("🧩 Stacking Model Architecture")
st.write("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression
""")

# ---------------- Prediction ----------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    lr_pred = classifier.named_estimators_["lr"].predict(input_scaled)[0]
    dt_pred = classifier.named_estimators_["dt"].predict(input_scaled)[0]
    rf_pred = classifier.named_estimators_["rf"].predict(input_scaled)[0]

    final_pred = classifier.predict(input_scaled)[0]
    confidence = classifier.predict_proba(input_scaled)[0][final_pred] * 100

    st.subheader("📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write("### 📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    st.write("### 🧠 Final Stacking Decision")
    st.write("Approved" if final_pred else "Rejected")

    st.write(f"### 📈 Confidence Score: {confidence:.2f}%")

    st.subheader("💼 Business Explanation")
    if final_pred == 1:
        st.info("Based on income, credit history, and combined model predictions, the applicant is likely to repay the loan. Therefore, the loan is approved.")
    else:
        st.info("Based on income, credit history, and combined model predictions, the applicant is unlikely to repay the loan. Therefore, the loan is rejected.")
        

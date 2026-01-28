import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Create Synthetic Loan Dataset
# -----------------------------
np.random.seed(42)
n = 1500

df = pd.DataFrame({
    "ApplicantIncome": np.random.randint(1500, 25000, n),
    "CoapplicantIncome": np.random.randint(0, 15000, n),
    "LoanAmount": np.random.randint(50, 1500, n),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], n),
    "Credit_History": np.random.choice([0, 1], n, p=[0.3, 0.7]),
    "Employment_Status": np.random.choice([0, 1], n, p=[0.4, 0.6]),
    "Property_Area": np.random.choice([0, 1, 2], n),
})

# -----------------------------
# REALISTIC, BALANCED TARGET LOGIC
# -----------------------------
approval_score = (
    (df["ApplicantIncome"] > 6000).astype(int) +
    (df["CoapplicantIncome"] > 3000).astype(int) +
    (df["Credit_History"] == 1).astype(int) +
    (df["LoanAmount"] < (df["ApplicantIncome"] + df["CoapplicantIncome"]) * 0.4).astype(int) +
    (df["Loan_Amount_Term"] >= 240).astype(int)
)

df["Loan_Status"] = (approval_score >= 4).astype(int)

print("\nLoan Status Distribution:")
print(df["Loan_Status"].value_counts())

# -----------------------------
# Split
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling (IMPORTANT FIX)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Base Models
# -----------------------------
lr = LogisticRegression(max_iter=5000)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# -----------------------------
# Stacking Model
# -----------------------------
stacking_model = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    final_estimator=LogisticRegression(max_iter=3000),
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation (Sanity Check)
# -----------------------------
y_pred = stacking_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Stacking Model Accuracy: {acc:.2f}")

# -----------------------------
# SAVE FILES
# -----------------------------
joblib.dump(stacking_model, "stacking_loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("\n✅ All PKL files created successfully")

from sklearn.metrics import classification_report

X_test_scaled = scaler.transform(X_test)
y_pred = stacking_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

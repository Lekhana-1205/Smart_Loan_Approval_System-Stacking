import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# -----------------------------
# Create Synthetic Loan Dataset
# -----------------------------
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "ApplicantIncome": np.random.randint(1500, 25000, n),
    "CoapplicantIncome": np.random.randint(0, 15000, n),
    "LoanAmount": np.random.randint(50, 700, n),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], n),
    "Credit_History": np.random.choice([0, 1], n),
    "Employment_Status": np.random.choice([0, 1], n),
    "Property_Area": np.random.choice([0, 1, 2], n),
})

df["Loan_Status"] = (
    (df["ApplicantIncome"] > 4000) &
    (df["Credit_History"] == 1) &
    (df["LoanAmount"] < 500)
).astype(int)

# -----------------------------
# Split
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------
# Models
# -----------------------------
lr = LogisticRegression(max_iter=5000)
dt = DecisionTreeClassifier(max_depth=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

stacking_model = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_model.fit(X_train_scaled, y_train)

# -----------------------------
# SAVE FILES (IMPORTANT)
# -----------------------------
joblib.dump(stacking_model, "stacking_loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("✅ All PKL files created successfully")

#SmartPremium Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

#Define FeatureEngineer (same as training)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.risk_map = {"House": 3, "Condo": 2, "Apartment": 1}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def cap_outliers(series, lower=0.01, upper=0.99):
            lower_bound = series.quantile(lower)
            upper_bound = series.quantile(upper)
            return np.clip(series, lower_bound, upper_bound)

        if "Previous Claims" in X.columns:
            X["Previous Claims"] = cap_outliers(X["Previous Claims"])

        #Log transforms
        for col in ["Annual Income", "Health Score"]:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        #Binning
        if "Annual Income" in X.columns:
            X["Income_Band"] = pd.qcut(X["Annual Income"], q=5, labels=False, duplicates="drop")
        if "Health Score" in X.columns:
            X["Health_Band"] = pd.qcut(X["Health Score"], q=5, labels=False, duplicates="drop")
        if "Credit Score" in X.columns:
            X["Credit_Band"] = pd.qcut(X["Credit Score"], q=5, labels=False, duplicates="drop")

        #Age groups
        if "Age" in X.columns:
            X["Age_Group"] = pd.cut(X["Age"], bins=[18, 30, 45, 60, 80],
                                    labels=["Young", "Mid", "Mature", "Senior"])
        if "Smoking Status" in X.columns and "Age" in X.columns:
            X["Age_Smoking"] = X["Age"] * (X["Smoking Status"] == "Yes").astype(int)

        #Vehicle policy
        if "Vehicle Age" in X.columns and "Policy Type" in X.columns:
            X["VehiclePolicy"] = X["Vehicle Age"].astype(str) + "_" + X["Policy Type"].astype(str)

        #Duration categories
        if "Insurance Duration" in X.columns:
            X["Duration_Category"] = pd.cut(
                X["Insurance Duration"], bins=[0, 2, 5, 10, 20],
                labels=["Short", "Mid", "Long", "Very Long"]
            )

        #Claims per year
        if "Previous Claims" in X.columns and "Insurance Duration" in X.columns:
            X["Claims_per_Year"] = X["Previous Claims"] / (X["Insurance Duration"] + 1)

        #Property risk
        if "Property Type" in X.columns:
            X["Property_Risk"] = X["Property Type"].map(self.risk_map)

        return X

#Load trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="SmartPremium: Insurance Cost Predictor", layout="centered")
st.title("💡 SmartPremium: Predict Insurance Premiums")
st.markdown("Fill in customer details to get a premium estimate.")

#Input Form
with st.form("prediction_form"):
    st.subheader("Customer Information")

    age = st.number_input("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income ($)", 1000, 200000, 60000, step=1000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents", 0, 10, 0)

    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])

    health_score = st.number_input("Health Score (0-100)", 0, 100, 70)
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])

    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    prev_claims = st.number_input("Previous Claims", 0, 20, 0)
    vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 5)
    credit_score = st.number_input("Credit Score", 300, 850, 700)
    duration = st.number_input("Insurance Duration (years)", 1, 20, 5)
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    submitted = st.form_submit_button("Predict Premium 💰")

#Prediction
if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": prev_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": duration,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise,
        "Property Type": property_type,
        "Policy_Start_Year": 2025,
        "Policy_Start_Month": 1
    }])

    prediction = model.predict(input_data)[0]
    premium = np.expm1(prediction)  #inverse of log1p

    st.success(f"💰 Predicted Insurance Premium: **${premium:,.2f}**")

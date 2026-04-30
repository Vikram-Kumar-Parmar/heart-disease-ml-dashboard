import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load artifacts
#-------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

#-------------------------

scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction App")

st.write("Enter patient details below:")

# Input fields
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.selectbox("Rest ECG (0–2)", [0,1,2])
thalach = st.slider("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0,1,2])
ca = st.selectbox("Major Vessels (0–3)", [0,1,2,3])
thal = st.selectbox("Thal (1–3)", [1,2,3])

# Convert input to dataframe
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg,
    "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
    "slope": slope, "ca": ca, "thal": thal
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding (same as training)
input_df = pd.get_dummies(input_df)

# Align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale
num_cols = ["age","trestbps","chol","thalach","oldpeak","ca"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"Disease Present (Confidence: {prob:.2f})")
    else:
        st.success(f"No Disease (Confidence: {1-prob:.2f})")

    # Feature importance (top 3)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=columns).sort_values(ascending=False)

    st.subheader("Top Influencing Features")
    st.bar_chart(feat_imp.head(3))

    st.write(
    "The model predicts risk based on patterns seen in similar patients. "
    "In this case, the highlighted features contributed most strongly to the decision."
)

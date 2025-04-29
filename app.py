import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature list
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("feature_columns.txt", "r") as f:
    feature_columns = f.read().split(",")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction App")

st.markdown("Use the sliders, checkboxes, and options below to input patient data.")

def user_input():
    age = st.slider("Age", 20, 90, 50)
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)

    # Gender
    sex = st.radio("Sex", ["Male", "Female"])
    sex_male = 1 if sex == "Male" else 0

    fasting_bs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")
    exercise_angina = st.checkbox("Exercise-Induced Angina")

    # Chest Pain Type
    cp_type = st.radio("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    cp_map = {
        "Typical Angina": "ChestPainType_TA",
        "Atypical Angina": "ChestPainType_ATA",
        "Non-anginal Pain": "ChestPainType_NAP",
        "Asymptomatic": "ChestPainType_ASY"
    }

    # Resting ECG
    restecg = st.radio("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    ecg_map = {
        "Normal": "RestingECG_Normal",
        "ST-T Abnormality": "RestingECG_ST",
        "Left Ventricular Hypertrophy": "RestingECG_LVH"
    }

    # ST Slope
    slope = st.radio("ST Slope", ["Up", "Flat", "Down"])
    slope_map = {
        "Up": "ST_Slope_Up",
        "Flat": "ST_Slope_Flat",
        "Down": "ST_Slope_Down"
    }

    # Thalassemia
    thal = st.radio("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    thal_map = {
        "Normal": "Thal_Normal",
        "Fixed Defect": "Thal_Fixed",
        "Reversible Defect": "Thal_Reversible"
    }

    # Create a dictionary for the input features
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_Male": sex_male,
        "FastingBS": int(fasting_bs),
        "ExerciseAngina": int(exercise_angina),
    }

    # Add one-hot encodings
    for col in feature_columns:
        if col.startswith("ChestPainType_"):
            input_data[col] = 1 if col == cp_map[cp_type] else 0
        elif col.startswith("RestingECG_"):
            input_data[col] = 1 if col == ecg_map[restecg] else 0
        elif col.startswith("ST_Slope_"):
            input_data[col] = 1 if col == slope_map[slope] else 0
        elif col.startswith("Thal_"):
            input_data[col] = 1 if col == thal_map[thal] else 0
        elif col not in input_data:
            input_data[col] = 0  # fill missing expected one-hot encodings with 0

    return pd.DataFrame([input_data])

# Get user input
data = user_input()

# Scale numeric features
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
data[numeric_cols] = scaler.transform(data[numeric_cols])

# Predict
if st.button("Predict Heart Disease"):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of Heart Disease. Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low risk of Heart Disease. Probability: {(1-probability)*100:.2f}%")

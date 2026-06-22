import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Coronary Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =====================================
# LOAD MODEL FILES
# =====================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("Utils/random_forest.pkl")

    scaler = joblib.load("Utils/scaler.pkl")

    feature_columns = joblib.load(
        "Utils/feature_columns.pkl"
    )

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

# =====================================
# HEADER
# =====================================

st.title("Coronary Heart Disease Prediction System")

st.markdown("""
This system predicts whether a patient is at risk of Coronary Heart Disease (CHD)
using a Random Forest Machine Learning model trained on clinical cardiovascular data.

The prediction is intended as a decision-support tool and not a replacement for professional medical diagnosis.
""")

# =====================================
# INPUT SECTION
# =====================================

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "asymptomatic",
            "atypical angina",
            "non-anginal",
            "typical angina"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [False, True]
    )

with col2:

    restecg = st.selectbox(
        "Resting ECG",
        [
            "lv hypertrophy",
            "normal",
            "st-t abnormality"
        ]
    )

    thalch = st.number_input(
        "Maximum Heart Rate Achieved",
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [False, True]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        value=1.0,
        step=0.1
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    slope = st.selectbox(
        "Slope",
        [
            "downsloping",
            "flat",
            "upsloping"
        ]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "fixed defect",
            "normal",
            "reversable defect"
        ]
    )

# =====================================
# BUILD INPUT DATA
# =====================================

def create_input_dataframe():

    data = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": int(fbs),
        "thalch": thalch,
        "exang": int(exang),
        "oldpeak": oldpeak,
        "ca": ca,
        "sex_Male": 1 if sex == "Male" else 0,
        "cp_atypical angina": 1 if cp == "atypical angina" else 0,
        "cp_non-anginal": 1 if cp == "non-anginal" else 0,
        "cp_typical angina": 1 if cp == "typical angina" else 0,
        "restecg_normal": 1 if restecg == "normal" else 0,
        "restecg_st-t abnormality": 1 if restecg == "st-t abnormality" else 0,
        "slope_flat": 1 if slope == "flat" else 0,
        "slope_upsloping": 1 if slope == "upsloping" else 0,
        "thal_normal": 1 if thal == "normal" else 0,
        "thal_reversable defect": 1 if thal == "reversable defect" else 0
    }

    input_df = pd.DataFrame([data])

    for col in feature_columns:

        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    return input_df

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Heart Disease Risk"):

    input_df = create_input_dataframe()

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(
        scaled_input
    )[0]

    probability = model.predict_proba(
        scaled_input
    )[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"Heart Disease Detected\n\nRisk Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"No Heart Disease Detected\n\nConfidence: {(1-probability):.2%}"
        )

    st.metric(
        "Disease Probability",
        f"{probability:.2%}"
    )

# =====================================
# MODEL INFO
# =====================================

st.markdown("---")

st.header("Model Information")

st.write("""
Model Used: Random Forest Classifier

Performance During Evaluation:

• Accuracy: 84.24%

• Precision: 82.88%

• Recall: 90.20%

• F1 Score: 86.39%

• ROC-AUC: 90.78%

The Random Forest model achieved the best overall performance among all evaluated models and was therefore selected for deployment.
""")
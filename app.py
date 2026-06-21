import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Coronary Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    scaler = joblib.load("Models/scaler.pkl")

    lr_model = joblib.load("Models/logistic_regression.pkl")

    svm_model = joblib.load("Models/svm.pkl")

    rf_model = joblib.load("Models/random_forest.pkl")

    ann_model = tf.keras.models.load_model(
        "Models/ann_model.keras"
    )

    return scaler, lr_model, svm_model, rf_model, ann_model


scaler, lr_model, svm_model, rf_model, ann_model = load_models()


# ==================================================
# APP TITLE
# ==================================================

st.title("❤️ Coronary Heart Disease Prediction System")

st.markdown("""
This system predicts the likelihood of Coronary Heart Disease (CHD)
using Machine Learning models trained on clinical patient data.

Available Models:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Artificial Neural Network (ANN)

Random Forest achieved the highest performance during evaluation.
""")


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "ANN"
    ]
)


# ==================================================
# INPUT FORM
# ==================================================

st.subheader("Patient Information")

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
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol",
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["False", "True"]
    )

with col2:

    restecg = st.selectbox(
        "Resting ECG",
        [
            "normal",
            "st-t abnormality",
            "lv hypertrophy"
        ]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["False", "True"]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [
            "upsloping",
            "flat",
            "downsloping"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "normal",
            "fixed defect",
            "reversible defect"
        ]
    )


# ==================================================
# ENCODING
# ==================================================

def prepare_input():

    sex_map = {
        "Male": 1,
        "Female": 0
    }

    cp_map = {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal": 2,
        "asymptomatic": 3
    }

    fbs_map = {
        "False": 0,
        "True": 1
    }

    restecg_map = {
        "normal": 0,
        "st-t abnormality": 1,
        "lv hypertrophy": 2
    }

    exang_map = {
        "False": 0,
        "True": 1
    }

    slope_map = {
        "upsloping": 0,
        "flat": 1,
        "downsloping": 2
    }
    thal_map = {
        "normal": 0,
        "fixed defect": 1,
        "reversible defect": 2
    }

    data = pd.DataFrame({
        "age": [age],
        "sex": [sex_map[sex]],
        "cp": [cp_map[cp]],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs_map[fbs]],
        "restecg": [restecg_map[restecg]],
        "thalch": [thalach],
        "exang": [exang_map[exang]],
        "oldpeak": [oldpeak],
        "slope": [slope_map[slope]],
        "ca": [ca],
        "thal": [thal_map[thal]]
    })

    return data


# ==================================================
# PREDICTION
# ==================================================

if st.button("Predict"):

    patient_data = prepare_input()

    scaled_data = scaler.transform(patient_data)

    if model_choice == "Random Forest":

        probability = rf_model.predict_proba(
            scaled_data
        )[0][1]

        prediction = rf_model.predict(
            scaled_data
        )[0]

    elif model_choice == "Logistic Regression":

        probability = lr_model.predict_proba(
            scaled_data
        )[0][1]

        prediction = lr_model.predict(
            scaled_data
        )[0]

    elif model_choice == "SVM":

        probability = svm_model.predict_proba(
            scaled_data
        )[0][1]

        prediction = svm_model.predict(
            scaled_data
        )[0]

    else:

        probability = ann_model.predict(
            scaled_data,
            verbose=0
        )[0][0]

        prediction = int(probability > 0.5)

    st.markdown("---")

    if prediction == 1:

        st.error(
            f"Heart Disease Detected\n\nRisk Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"No Heart Disease Detected\n\nRisk Probability: {(1-probability):.2%}"
        )

    st.subheader("Prediction Summary")

    result_df = pd.DataFrame({
        "Model Used": [model_choice],
        "Prediction": [
            "Heart Disease"
            if prediction == 1
            else "No Heart Disease"
        ],
        "Probability": [round(probability, 4)]
    })

    st.dataframe(
        result_df,
        use_container_width=True
    )


# # ==================================================
# # MODEL PERFORMANCE
# # ==================================================

# st.markdown("---")

# st.subheader("Model Performance Summary")

# performance_df = pd.DataFrame({
#     "Model": [
#         "Logistic Regression",
#         "SVM",
#         "Random Forest",
#         "ANN"
#     ],
#     "Accuracy": [
#         0.8043,
#         0.8261,
#         0.8424,
#         0.8315
#     ],
#     "ROC AUC": [
#         0.8926,
#         0.9067,
#         0.9078,
#         0.8993
#     ]
# })

# st.dataframe(
#     performance_df,
#     use_container_width=True
# )

# st.info(
#     "Random Forest achieved the highest overall performance and is recommended for deployment."
# )
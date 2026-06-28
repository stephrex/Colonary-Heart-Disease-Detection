"""
Coronary Heart Disease Prediction — Multi-Model Inference Dashboard
-------------------------------------------------------------------
Loads the trained models (Logistic Regression, SVM, Random Forest, ANN)
and the StandardScaler produced in the modelling notebook, and exposes:

  1. Single-patient inference  (interactive form)
  2. Batch inference            (CSV upload)
  3. Model insights             (performance metrics + evaluation plots)

Default / recommended model: Random Forest (highest evaluation score).
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent
UTILS_DIR = PROJECT_ROOT / "Utils"
PLOTS_DIR = PROJECT_ROOT / "Plots"
EVAL_PLOTS_DIR = PLOTS_DIR / "Evaluation_Plots"
DATA_DIR = PROJECT_ROOT / "Dataset" / "Processed Datasets"

# Feature order must match the scaler's fit-time column order
# (cleaned_heart_disease_data.csv with `num` and `target` dropped).
FEATURE_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "fbs",
    "thalch",
    "exang",
    "oldpeak",
    "ca",
    "sex_Male",
    "cp_atypical angina",
    "cp_non-anginal",
    "cp_typical angina",
    "restecg_normal",
    "restecg_st-t abnormality",
    "slope_flat",
    "slope_upsloping",
    "thal_normal",
    "thal_reversable defect",
]

MODEL_FILES = {
    "Random Forest": ("random_forest_model.pkl", "sklearn"),
    "Logistic Regression": ("logistic_regression_model.pkl", "sklearn"),
    "SVM": ("svm_model.pkl", "sklearn"),
    "ANN": ("ann_model.h5", "keras"),
}

RECOMMENDED_MODEL = "Random Forest"

# ============================================================
# PAGE CONFIG + STYLE
# ============================================================

st.set_page_config(
    page_title="Coronary Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div { padding-top: 1.2rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 18px;
        border-radius: 14px;
        color: #fff;
    }
    div[data-testid="stMetricLabel"] { color: #c9c9d4 !important; }
    .pred-card {
        background: linear-gradient(135deg, #11151c 0%, #1c2230 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 6px 22px rgba(0,0,0,0.25);
    }
    .pred-card.risk    { border-left: 5px solid #ff4d6d; }
    .pred-card.safe    { border-left: 5px solid #2bd97c; }
    .pred-title        { font-size: 1.05rem; font-weight: 600; color: #ffffff; margin: 0; }
    .pred-sub          { color: #a8b0c2; font-size: 0.85rem; margin-top: 2px; }
    .pred-prob         { font-size: 2.0rem; font-weight: 700; margin-top: 6px; color: #fff; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 999px;
        margin-left: 6px;
        vertical-align: middle;
    }
    .badge.rec   { background: #2bd97c33; color: #2bd97c; border: 1px solid #2bd97c66; }
    .badge.risk  { background: #ff4d6d33; color: #ff4d6d; border: 1px solid #ff4d6d66; }
    .badge.safe  { background: #2bd97c33; color: #2bd97c; border: 1px solid #2bd97c66; }
    .hero {
        background: linear-gradient(135deg, #ff4d6d 0%, #c9184a 50%, #6a040f 100%);
        padding: 22px 28px;
        border-radius: 18px;
        color: #fff;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(201,24,74,0.35);
    }
    .hero h1 { margin: 0 0 4px 0; color: #fff; }
    .hero p  { margin: 0; opacity: 0.92; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ARTIFACT LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    return joblib.load(UTILS_DIR / "scaler.pkl")


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    filename, kind = MODEL_FILES[name]
    path = UTILS_DIR / filename
    if kind == "sklearn":
        return joblib.load(path)
    if kind == "keras":
        # Lazy import — only needed if the ANN model is selected.
        from tensorflow.keras.models import load_model as keras_load_model
        return keras_load_model(path, compile=False)
    raise ValueError(f"Unknown model kind: {kind}")


@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = UTILS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def predict_with_model(name: str, X_scaled: np.ndarray):
    """Returns (predictions, probabilities) regardless of model type."""
    model = load_model(name)
    _, kind = MODEL_FILES[name]
    if kind == "keras":
        prob = model.predict(X_scaled, verbose=0).flatten()
        pred = (prob >= 0.5).astype(int)
        return pred, prob
    prob = model.predict_proba(X_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


# ============================================================
# INPUT BUILDERS
# ============================================================

def build_single_input(form_values: dict) -> pd.DataFrame:
    """Build a 1-row DataFrame matching FEATURE_COLUMNS exactly."""
    fv = form_values
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["age"] = fv["age"]
    row["trestbps"] = fv["trestbps"]
    row["chol"] = fv["chol"]
    row["fbs"] = int(fv["fbs"])
    row["thalch"] = fv["thalch"]
    row["exang"] = int(fv["exang"])
    row["oldpeak"] = fv["oldpeak"]
    row["ca"] = fv["ca"]
    row["sex_Male"] = 1 if fv["sex"] == "Male" else 0
    if fv["cp"] != "asymptomatic":
        row[f"cp_{fv['cp']}"] = 1
    if fv["restecg"] != "lv hypertrophy":
        row[f"restecg_{fv['restecg']}"] = 1
    if fv["slope"] != "downsloping":
        row[f"slope_{fv['slope']}"] = 1
    if fv["thal"] != "fixed defect":
        row[f"thal_{fv['thal']}"] = 1
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def coerce_batch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a batch CSV in the cleaned-dataset schema.
    Drops `num` / `target` if present and aligns to FEATURE_COLUMNS.
    Missing one-hot columns are added as 0.
    """
    df = df.copy()
    df.drop(columns=[c for c in ("num", "target") if c in df.columns], inplace=True)
    for col in ("fbs", "exang"):
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(int)
    bool_like = df.select_dtypes(include=["bool"]).columns
    for col in bool_like:
        df[col] = df[col].astype(int)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS]


# ============================================================
# SIDEBAR — MODEL SELECTION
# ============================================================

scaler = load_scaler()
comparison_df = load_model_comparison()

with st.sidebar:
    st.markdown("### Model Selection")
    st.caption(
        "Pick one or more models to run inference with. "
        f"**{RECOMMENDED_MODEL}** is selected by default — it had the highest evaluation score."
    )

    available_models = list(MODEL_FILES.keys())
    label_map = {
        m: (f"{m}  ⭐  (recommended)" if m == RECOMMENDED_MODEL else m)
        for m in available_models
    }

    selected_models = st.multiselect(
        "Models",
        options=available_models,
        default=[RECOMMENDED_MODEL],
        format_func=lambda m: label_map[m],
    )

    if not selected_models:
        st.warning("Select at least one model to enable inference.")

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This dashboard performs inference using models trained in the "
        "**Coronary Heart Disease Detection** research project."
    )
    st.caption("Decision-support tool — not a substitute for clinical diagnosis.")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>❤️ Coronary Heart Disease Prediction</h1>
        <p>Multi-model inference dashboard — single patient & batch predictions powered by the trained research models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_insights = st.tabs(
    ["🩺  Single Patient", "📂  Batch CSV", "📊  Model Insights"]
)


# ----------------------------------------------------------------
# TAB 1 — SINGLE PATIENT
# ----------------------------------------------------------------
with tab_single:
    st.subheader("Patient Information")
    st.caption("Enter the patient's clinical features, then run inference.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["asymptomatic", "atypical angina", "non-anginal", "typical angina"],
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])

    with col2:
        restecg = st.selectbox(
            "Resting ECG", ["lv hypertrophy", "normal", "st-t abnormality"]
        )
        thalch = st.number_input("Maximum Heart Rate Achieved", value=150)
        exang = st.selectbox("Exercise Induced Angina", [False, True])
        oldpeak = st.number_input("Oldpeak", value=1.0, step=0.1)
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        slope = st.selectbox("Slope", ["downsloping", "flat", "upsloping"])
        thal = st.selectbox(
            "Thalassemia", ["fixed defect", "normal", "reversable defect"]
        )

    st.markdown("")
    run_single = st.button(
        "🔮  Predict Heart Disease Risk",
        type="primary",
        use_container_width=True,
        disabled=not selected_models,
    )

    if run_single:
        form_values = dict(
            age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
            restecg=restecg, thalch=thalch, exang=exang, oldpeak=oldpeak,
            ca=ca, slope=slope, thal=thal,
        )
        input_df = build_single_input(form_values)
        X_scaled = scaler.transform(input_df)

        st.markdown("---")
        st.subheader("Prediction Results")

        # Cards row
        card_cols = st.columns(min(len(selected_models), 4))
        per_model = []
        for i, name in enumerate(selected_models):
            pred, prob = predict_with_model(name, X_scaled)
            p = float(prob[0])
            label = int(pred[0])
            per_model.append((name, label, p))

            with card_cols[i % len(card_cols)]:
                risk_class = "risk" if label == 1 else "safe"
                outcome = "Heart Disease Detected" if label == 1 else "No Heart Disease"
                badge = (
                    '<span class="badge risk">AT RISK</span>'
                    if label == 1
                    else '<span class="badge safe">LOW RISK</span>'
                )
                rec_badge = (
                    '<span class="badge rec">recommended</span>'
                    if name == RECOMMENDED_MODEL
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="pred-card {risk_class}">
                        <p class="pred-title">{name}{rec_badge}</p>
                        <p class="pred-sub">{outcome} {badge}</p>
                        <p class="pred-prob">{p:.1%}</p>
                        <p class="pred-sub">Disease probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Probability gauges
        st.markdown("##### Probability Gauges")
        gauge_cols = st.columns(min(len(per_model), 4))
        for i, (name, label, p) in enumerate(per_model):
            with gauge_cols[i % len(gauge_cols)]:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=p * 100,
                        number={"suffix": "%", "font": {"size": 28}},
                        title={"text": name, "font": {"size": 14}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#ff4d6d" if label == 1 else "#2bd97c"},
                            "steps": [
                                {"range": [0, 50], "color": "rgba(43,217,124,0.15)"},
                                {"range": [50, 100], "color": "rgba(255,77,109,0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                fig.update_layout(
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e6e6f0"},
                )
                st.plotly_chart(fig, use_container_width=True)

        # Cross-model comparison
        if len(per_model) > 1:
            st.markdown("##### Cross-Model Comparison")
            cmp_df = pd.DataFrame(
                [
                    {
                        "Model": n,
                        "Prediction": "At Risk" if l == 1 else "Low Risk",
                        "Disease Probability": p,
                    }
                    for n, l, p in per_model
                ]
            )
            fig = px.bar(
                cmp_df,
                x="Model",
                y="Disease Probability",
                color="Prediction",
                color_discrete_map={"At Risk": "#ff4d6d", "Low Risk": "#2bd97c"},
                text=cmp_df["Disease Probability"].apply(lambda v: f"{v:.1%}"),
                range_y=[0, 1],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6e6f0"},
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------
# TAB 2 — BATCH CSV
# ----------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Inference from CSV")
    st.caption(
        "Upload a CSV in the cleaned-dataset schema "
        "(same columns as `Dataset/Processed Datasets/cleaned_heart_disease_data.csv`). "
        "Optional `target` / `num` columns are ignored for inference."
    )

    sample_path = DATA_DIR / "cleaned_heart_disease_data.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button(
                "⬇️  Download sample CSV template",
                data=f,
                file_name="cleaned_heart_disease_sample.csv",
                mime="text/csv",
            )

    uploaded = st.file_uploader("Upload patient batch CSV", type=["csv"])

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        st.markdown("**Preview — uploaded data**")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"{len(raw_df)} rows · {len(raw_df.columns)} columns")

        run_batch = st.button(
            "🚀  Run Batch Inference",
            type="primary",
            use_container_width=True,
            disabled=not selected_models,
        )

        if run_batch:
            try:
                aligned = coerce_batch_dataframe(raw_df)
            except Exception as e:
                st.error(f"Could not align CSV to model schema: {e}")
            else:
                X_scaled = scaler.transform(aligned)
                result_df = raw_df.copy().reset_index(drop=True)

                summary_rows = []
                for name in selected_models:
                    pred, prob = predict_with_model(name, X_scaled)
                    result_df[f"{name} — Probability"] = np.round(prob, 4)
                    result_df[f"{name} — Prediction"] = np.where(
                        pred == 1, "At Risk", "Low Risk"
                    )
                    summary_rows.append(
                        {
                            "Model": name,
                            "At Risk": int((pred == 1).sum()),
                            "Low Risk": int((pred == 0).sum()),
                            "Avg Probability": float(prob.mean()),
                        }
                    )

                st.markdown("---")
                st.subheader("Inference Summary")
                summary_df = pd.DataFrame(summary_rows)
                m_cols = st.columns(len(summary_rows))
                for i, row in enumerate(summary_rows):
                    with m_cols[i]:
                        st.metric(
                            label=row["Model"],
                            value=f"{row['At Risk']} at risk",
                            delta=f"{row['Avg Probability']:.1%} avg prob",
                        )

                fig = px.bar(
                    summary_df.melt(
                        id_vars="Model",
                        value_vars=["At Risk", "Low Risk"],
                        var_name="Class",
                        value_name="Count",
                    ),
                    x="Model",
                    y="Count",
                    color="Class",
                    barmode="group",
                    color_discrete_map={"At Risk": "#ff4d6d", "Low Risk": "#2bd97c"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e6e6f0"},
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### Per-row predictions")
                st.dataframe(result_df, use_container_width=True, height=420)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download predictions as CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ----------------------------------------------------------------
# TAB 3 — MODEL INSIGHTS
# ----------------------------------------------------------------
with tab_insights:
    st.subheader("Model Performance")

    if comparison_df is not None:
        styled = comparison_df.copy()
        for col in styled.columns:
            if col != "Model":
                styled[col] = styled[col].astype(float)
        st.dataframe(
            styled.style.format({c: "{:.2%}" for c in styled.columns if c != "Model"})
                        .background_gradient(
                            subset=[c for c in styled.columns if c != "Model"],
                            cmap="RdYlGn",
                        ),
            use_container_width=True,
        )

        metric_cols = [c for c in comparison_df.columns if c != "Model"]
        long_df = comparison_df.melt(
            id_vars="Model", value_vars=metric_cols,
            var_name="Metric", value_name="Score",
        )
        fig = px.bar(
            long_df,
            x="Model", y="Score", color="Metric",
            barmode="group", range_y=[0, 1],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6e6f0"},
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model comparison file not found.")

    st.markdown("---")
    st.subheader("Evaluation Plots")

    plot_options = {
        "Model comparison (bar)": EVAL_PLOTS_DIR / "model_comparison_with_values.png",
        "Model performance heatmap": EVAL_PLOTS_DIR / "model_performance_heatmap.png",
        "ROC curves": EVAL_PLOTS_DIR / "roc_curve_comparison.png",
        "Precision-Recall curves": EVAL_PLOTS_DIR / "precision_recall_curve.png",
        "Random Forest feature importance": EVAL_PLOTS_DIR / "random_forest_feature_importance.png",
        "Logistic Regression — confusion matrix": EVAL_PLOTS_DIR / "Logistic Regression_confusion_matrix.png",
        "SVM — confusion matrix": EVAL_PLOTS_DIR / "SVM_confusion_matrix.png",
        "Random Forest — confusion matrix": EVAL_PLOTS_DIR / "Random Forest_confusion_matrix.png",
        "ANN — confusion matrix": EVAL_PLOTS_DIR / "ANN_confusion_matrix.png",
    }
    available = {k: v for k, v in plot_options.items() if v.exists()}

    if available:
        choice = st.selectbox("Choose a plot to view", list(available.keys()))
        st.image(str(available[choice]), use_container_width=True)
    else:
        st.info("No evaluation plots found.")

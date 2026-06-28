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

# Refined, calm palette.
COLOR_RISK = "#B91C1C"   # deep calm red
COLOR_SAFE = "#059669"   # calm emerald
COLOR_INK = "#202123"
COLOR_MUTED = "#6B7280"
COLOR_BORDER = "#E5E7EB"
COLOR_SURFACE = "#FFFFFF"
COLOR_SURFACE_2 = "#F7F7F8"


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


TF_AVAILABLE = _tensorflow_available()


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
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                     "Helvetica Neue", Arial, sans-serif;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.015em;
        color: #202123;
        font-weight: 600;
    }
    h1 { font-size: 1.9rem; margin-bottom: 0.2rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.05rem; }

    /* Header */
    .page-header {
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 1.2rem;
        margin-bottom: 1.6rem;
    }
    .page-header .title-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .page-header .heart { font-size: 1.6rem; line-height: 1; }
    .page-header h1 { margin: 0; font-size: 1.7rem; }
    .page-header p.subtitle {
        color: #6B7280;
        font-size: 0.95rem;
        margin: 6px 0 0 0;
        max-width: 720px;
    }

    /* Tabs — clean underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 14px;
        background: transparent;
        color: #6B7280;
        font-weight: 500;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #202123 !important;
        border-bottom: 2px solid #B91C1C !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        border: 1px solid #E5E7EB;
        padding: 0.55rem 1rem;
    }
    .stButton > button[kind="primary"] {
        background: #B91C1C;
        border-color: #B91C1C;
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover {
        background: #991B1B;
        border-color: #991B1B;
    }
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        font-weight: 500;
        background: #FFFFFF;
        color: #202123;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px !important;
        border-color: #E5E7EB !important;
    }
    .stNumberInput input {
        border-radius: 10px;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .card-accent-risk { border-left: 3px solid #B91C1C; }
    .card-accent-safe { border-left: 3px solid #059669; }
    .card .model-name {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0;
    }
    .card .verdict {
        font-size: 1.1rem;
        font-weight: 600;
        color: #202123;
        margin: 6px 0 0 0;
    }
    .card .prob {
        font-size: 2rem;
        font-weight: 600;
        color: #202123;
        margin: 10px 0 0 0;
        font-variant-numeric: tabular-nums;
    }
    .card .prob-label {
        font-size: 0.78rem;
        color: #6B7280;
        margin: 0;
    }
    .tag-rec {
        display: inline-block;
        margin-left: 8px;
        font-size: 0.65rem;
        font-weight: 600;
        color: #6B7280;
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 2px 8px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        vertical-align: middle;
    }

    /* Metric tweaks */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        font-weight: 500;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #202123;
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #F7F7F8;
        border-right: 1px solid #E5E7EB;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6B7280;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    /* Subtle section labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6B7280;
        margin: 1.4rem 0 0.6rem 0;
    }
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


def _build_ann_architecture(n_features: int):
    """
    Recreates the ANN topology used in the modelling notebook.
    Loading weights into a freshly-built architecture is robust to Keras
    version drift (the .h5 may have been saved with a newer Keras whose
    config kwargs an older Keras can't parse).
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    return model


def _load_keras_model(path: Path):
    # Strategy 1: native load_model (works when Keras versions match)
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        return keras_load_model(str(path), compile=False)
    except Exception:
        pass
    # Strategy 2: rebuild architecture, then load weights only
    model = _build_ann_architecture(len(FEATURE_COLUMNS))
    model.load_weights(str(path))
    return model


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    filename, kind = MODEL_FILES[name]
    path = UTILS_DIR / filename
    try:
        if kind == "sklearn":
            return joblib.load(path)
        if kind == "keras":
            return _load_keras_model(path)
    except Exception:
        return None
    return None


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
# SIDEBAR
# ============================================================

scaler = load_scaler()
comparison_df = load_model_comparison()

with st.sidebar:
    st.markdown("### Models")
    st.caption(
        "Select one or more models for inference. "
        f"{RECOMMENDED_MODEL} is the highest-scoring model and is selected by default."
    )

    candidate_models = [
        m for m in MODEL_FILES.keys()
        if MODEL_FILES[m][1] != "keras" or TF_AVAILABLE
    ]
    available_models = [m for m in candidate_models if load_model(m) is not None]
    unavailable_models = [m for m in candidate_models if m not in available_models]

    if not TF_AVAILABLE:
        st.caption("ANN is unavailable in this environment (TensorFlow not installed).")
    if unavailable_models:
        st.caption(
            "Unavailable: " + ", ".join(unavailable_models)
            + " (model file could not be loaded)."
        )

    label_map = {
        m: (f"{m}  ·  recommended" if m == RECOMMENDED_MODEL else m)
        for m in available_models
    }
    selected_models = st.multiselect(
        "Active models",
        options=available_models,
        default=[RECOMMENDED_MODEL],
        format_func=lambda m: label_map[m],
        label_visibility="collapsed",
    )

    if not selected_models:
        st.warning("Select at least one model to enable inference.")

    st.markdown("### About")
    st.write(
        "Inference dashboard for the Coronary Heart Disease Detection "
        "research project. Decision-support tool — not a clinical diagnosis."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="page-header">
        <div class="title-row">
            <span class="heart">❤️</span>
            <h1>Coronary Heart Disease Prediction</h1>
        </div>
        <p class="subtitle">
            Multi-model inference dashboard for the trained research models.
            Run single-patient or batch predictions, and review model performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_insights = st.tabs(
    ["Single Patient", "Batch CSV", "Model Insights"]
)


# ----------------------------------------------------------------
# TAB 1 — SINGLE PATIENT
# ----------------------------------------------------------------
with tab_single:
    st.markdown('<div class="section-label">Patient information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest pain type",
            ["asymptomatic", "atypical angina", "non-anginal", "typical angina"],
        )
        trestbps = st.number_input("Resting blood pressure (mm Hg)", value=120)
        chol = st.number_input("Serum cholesterol (mg/dl)", value=200)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [False, True])
    with col2:
        restecg = st.selectbox(
            "Resting ECG", ["lv hypertrophy", "normal", "st-t abnormality"]
        )
        thalch = st.number_input("Maximum heart rate achieved", value=150)
        exang = st.selectbox("Exercise-induced angina", [False, True])
        oldpeak = st.number_input("Oldpeak", value=1.0, step=0.1)
        ca = st.selectbox("Number of major vessels", [0, 1, 2, 3])
        slope = st.selectbox("Slope", ["downsloping", "flat", "upsloping"])
        thal = st.selectbox(
            "Thalassemia", ["fixed defect", "normal", "reversable defect"]
        )

    st.write("")
    run_single = st.button(
        "Predict heart disease risk",
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

        st.markdown('<div class="section-label">Predictions</div>', unsafe_allow_html=True)

        card_cols = st.columns(min(len(selected_models), 4))
        per_model = []
        for i, name in enumerate(selected_models):
            pred, prob = predict_with_model(name, X_scaled)
            p = float(prob[0])
            label = int(pred[0])
            per_model.append((name, label, p))

            with card_cols[i % len(card_cols)]:
                accent = "card-accent-risk" if label == 1 else "card-accent-safe"
                verdict = "At risk" if label == 1 else "Low risk"
                rec_tag = '<span class="tag-rec">recommended</span>' if name == RECOMMENDED_MODEL else ""
                st.markdown(
                    f"""
                    <div class="card {accent}">
                        <p class="model-name">{name}{rec_tag}</p>
                        <p class="verdict">{verdict}</p>
                        <p class="prob">{p:.1%}</p>
                        <p class="prob-label">Disease probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-label">Probability gauges</div>', unsafe_allow_html=True)
        gauge_cols = st.columns(min(len(per_model), 4))
        for i, (name, label, p) in enumerate(per_model):
            with gauge_cols[i % len(gauge_cols)]:
                bar_color = COLOR_RISK if label == 1 else COLOR_SAFE
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=p * 100,
                        number={"suffix": "%", "font": {"size": 26, "color": COLOR_INK}},
                        title={"text": name, "font": {"size": 13, "color": COLOR_MUTED}},
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": COLOR_MUTED,
                                "tickfont": {"color": COLOR_MUTED, "size": 10},
                            },
                            "bar": {"color": bar_color, "thickness": 0.25},
                            "bgcolor": COLOR_SURFACE_2,
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 50], "color": "#ECFDF5"},
                                {"range": [50, 100], "color": "#FEF2F2"},
                            ],
                            "threshold": {
                                "line": {"color": COLOR_MUTED, "width": 2},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                fig.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": COLOR_INK},
                )
                st.plotly_chart(fig, use_container_width=True)

        if len(per_model) > 1:
            st.markdown('<div class="section-label">Cross-model comparison</div>', unsafe_allow_html=True)
            cmp_df = pd.DataFrame(
                [
                    {
                        "Model": n,
                        "Prediction": "At risk" if l == 1 else "Low risk",
                        "Disease probability": p,
                    }
                    for n, l, p in per_model
                ]
            )
            fig = px.bar(
                cmp_df,
                x="Model",
                y="Disease probability",
                color="Prediction",
                color_discrete_map={"At risk": COLOR_RISK, "Low risk": COLOR_SAFE},
                text=cmp_df["Disease probability"].apply(lambda v: f"{v:.1%}"),
                range_y=[0, 1],
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLOR_INK, "family": "Inter, sans-serif"},
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor=COLOR_BORDER, tickformat=".0%"),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------
# TAB 2 — BATCH CSV
# ----------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="section-label">Batch inference from CSV</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a CSV in the cleaned-dataset schema "
        "(same columns as Dataset/Processed Datasets/cleaned_heart_disease_data.csv). "
        "Optional target and num columns are ignored for inference."
    )

    sample_path = DATA_DIR / "cleaned_heart_disease_data.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button(
                "Download sample CSV template",
                data=f,
                file_name="cleaned_heart_disease_sample.csv",
                mime="text/csv",
            )

    uploaded = st.file_uploader("Upload patient batch CSV", type=["csv"])

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"{len(raw_df)} rows  ·  {len(raw_df.columns)} columns")

        run_batch = st.button(
            "Run batch inference",
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
                        pred == 1, "At risk", "Low risk"
                    )
                    summary_rows.append(
                        {
                            "Model": name,
                            "At risk": int((pred == 1).sum()),
                            "Low risk": int((pred == 0).sum()),
                            "Avg probability": float(prob.mean()),
                        }
                    )

                st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
                summary_df = pd.DataFrame(summary_rows)
                m_cols = st.columns(len(summary_rows))
                for i, row in enumerate(summary_rows):
                    with m_cols[i]:
                        st.metric(
                            label=row["Model"],
                            value=f"{row['At risk']} at risk",
                            delta=f"{row['Avg probability']:.1%} avg probability",
                            delta_color="off",
                        )

                fig = px.bar(
                    summary_df.melt(
                        id_vars="Model",
                        value_vars=["At risk", "Low risk"],
                        var_name="Class",
                        value_name="Count",
                    ),
                    x="Model",
                    y="Count",
                    color="Class",
                    barmode="group",
                    color_discrete_map={"At risk": COLOR_RISK, "Low risk": COLOR_SAFE},
                )
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": COLOR_INK, "family": "Inter, sans-serif"},
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor=COLOR_BORDER),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-label">Per-row predictions</div>', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True, height=420)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ----------------------------------------------------------------
# TAB 3 — MODEL INSIGHTS
# ----------------------------------------------------------------
with tab_insights:
    st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)

    if comparison_df is not None:
        styled = comparison_df.copy()
        for col in styled.columns:
            if col != "Model":
                styled[col] = styled[col].astype(float)
        st.dataframe(
            styled.style.format({c: "{:.2%}" for c in styled.columns if c != "Model"})
                        .background_gradient(
                            subset=[c for c in styled.columns if c != "Model"],
                            cmap="Reds",
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
            color_discrete_sequence=["#B91C1C", "#DC2626", "#F87171", "#FCA5A5", "#FEE2E2"],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLOR_INK, "family": "Inter, sans-serif"},
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor=COLOR_BORDER, tickformat=".0%"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model comparison file not found.")

    st.markdown('<div class="section-label">Evaluation plots</div>', unsafe_allow_html=True)

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

"""
app.py
──────
Streamlit frontend for Cancer Risk Assessment.
• Works as a UI skeleton (demo mode) before model training
• Switches to live model inference once artifacts exist
Run: streamlit run app.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .main { background-color: #0f1117; }

  /* Risk badge */
  .risk-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .risk-low    { background: #1a3a1a; color: #4ade80; border: 2px solid #4ade80; }
  .risk-medium { background: #3a2a00; color: #fbbf24; border: 2px solid #fbbf24; }
  .risk-high   { background: #3a0000; color: #f87171; border: 2px solid #f87171; }

  /* Card-like section */
  .info-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid #4f8ef7;
  }
  .model-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

RISK_COLORS = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}
RISK_EMOJI  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

ARTIFACTS_READY = (
    os.path.exists("artifacts/models/xgboost_model.pkl") and
    os.path.exists("artifacts/models/random_forest_model.pkl") and
    os.path.exists("artifacts/preprocessors/scaler.pkl")
)


@st.cache_resource
def load_pipeline():
    """Load the trained pipeline (cached across reruns)."""
    from src.prediction_pipeline import PredictionPipeline
    return PredictionPipeline()


def mock_predict(input_dict: dict) -> dict:
    """
    Returns fake predictions for UI demo before model is trained.
    Deterministically based on Smoking + Genetic Risk score.
    """
    score = (
        input_dict.get("Smoking", 1)
        + input_dict.get("Genetic Risk", 1)
        + input_dict.get("Air Pollution", 1)
    ) / 3.0

    if score <= 2.5:
        label, probs = "Low",    {"Low": 0.78, "Medium": 0.17, "High": 0.05}
    elif score <= 5.5:
        label, probs = "Medium", {"Low": 0.15, "Medium": 0.68, "High": 0.17}
    else:
        label, probs = "High",   {"Low": 0.04, "Medium": 0.18, "High": 0.78}

    entry = {"label": label, "confidence": probs[label], "probabilities": probs}
    return {"xgboost": entry, "random_forest": entry}


def render_risk_badge(label: str) -> None:
    css_class = RISK_COLORS.get(label, "risk-low")
    st.markdown(
        f'<div class="risk-badge {css_class}">'
        f'{RISK_EMOJI.get(label, "")}  {label} Risk</div>',
        unsafe_allow_html=True,
    )


def render_prob_bar(probs: dict) -> None:
    cols = st.columns(3)
    colors = {"Low": "#4ade80", "Medium": "#fbbf24", "High": "#f87171"}
    for col, (cls, p) in zip(cols, probs.items()):
        with col:
            st.metric(cls, f"{p*100:.1f}%")
            st.progress(float(p))


# ── Sidebar — Input Form ──────────────────────────────────────────────────────

def sidebar_inputs() -> dict:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/cancer-ribbon.png",
            width=60,
        )
        st.title("Patient Parameters")
        st.caption("Fill in all fields and click **Predict**")
        st.divider()

        st.subheader("Demographics")
        age    = st.slider("Age", 14, 73, 35)
        gender = st.selectbox("Gender", ["Male (1)", "Female (2)"])
        gender_val = 1 if "Male" in gender else 2

        st.subheader("Environmental / Lifestyle")
        air_pollution       = st.slider("Air Pollution",       1, 8, 3)
        alcohol_use         = st.slider("Alcohol Use",         1, 8, 2)
        dust_allergy        = st.slider("Dust Allergy",        1, 8, 3)
        occ_hazards         = st.slider("Occupational Hazards",1, 8, 2)
        smoking             = st.slider("Smoking",             1, 8, 3)
        passive_smoker      = st.slider("Passive Smoker",      1, 8, 2)

        st.subheader("Medical History")
        genetic_risk        = st.slider("Genetic Risk",        1, 7, 3)
        chronic_lung        = st.slider("Chronic Lung Disease",1, 7, 2)
        balanced_diet       = st.slider("Balanced Diet",       1, 5, 3)
        obesity             = st.slider("Obesity",             1, 7, 3)

        st.subheader("Symptoms")
        chest_pain          = st.slider("Chest Pain",          1, 9, 2)
        coughing_blood      = st.slider("Coughing of Blood",   1, 9, 2)
        fatigue             = st.slider("Fatigue",             1, 9, 3)
        weight_loss         = st.slider("Weight Loss",         1, 8, 2)
        sob                 = st.slider("Shortness of Breath", 1, 9, 2)
        wheezing            = st.slider("Wheezing",            1, 8, 2)
        swallowing          = st.slider("Swallowing Difficulty",1, 8, 2)
        clubbing            = st.slider("Clubbing of Finger Nails",1, 9, 2)
        freq_cold           = st.slider("Frequent Cold",       1, 7, 2)
        dry_cough           = st.slider("Dry Cough",           1, 7, 2)
        snoring             = st.slider("Snoring",             1, 7, 2)

        predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

    return {
        "Age": age,
        "Gender": gender_val,
        "Air Pollution": air_pollution,
        "Alcohol use": alcohol_use,
        "Dust Allergy": dust_allergy,
        "OccuPational Hazards": occ_hazards,
        "Genetic Risk": genetic_risk,
        "chronic Lung Disease": chronic_lung,
        "Balanced Diet": balanced_diet,
        "Obesity": obesity,
        "Smoking": smoking,
        "Passive Smoker": passive_smoker,
        "Chest Pain": chest_pain,
        "Coughing of Blood": coughing_blood,
        "Fatigue": fatigue,
        "Weight Loss": weight_loss,
        "Shortness of Breath": sob,
        "Wheezing": wheezing,
        "Swallowing Difficulty": swallowing,
        "Clubbing of Finger Nails": clubbing,
        "Frequent Cold": freq_cold,
        "Dry Cough": dry_cough,
        "Snoring": snoring,
        "_predict": predict_btn,
    }


# ── Main Page ─────────────────────────────────────────────────────────────────

def main():
    st.title("🩺 Cancer Risk Assessment System")
    st.caption(
        "Multi-class risk classification using XGBoost and Random Forest "
        "with SMOTE/SMOTEENN class-imbalance handling."
    )

    if not ARTIFACTS_READY:
        st.warning(
            "⚠️  **Demo Mode** — Models not yet trained. "
            "Run `python -m src.pipeline` to train. "
            "Showing mock predictions for UI testing.",
            icon="🧪",
        )

    inputs = sidebar_inputs()
    predict_clicked = inputs.pop("_predict")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_predict, tab_compare, tab_about = st.tabs(
        ["🔮 Prediction", "📊 Model Comparison", "ℹ️ About"]
    )

    # ── Tab 1: Prediction ─────────────────────────────────────────────────────
    with tab_predict:
        if predict_clicked:
            with st.spinner("Running inference …"):
                if ARTIFACTS_READY:
                    pipeline = load_pipeline()
                    results  = pipeline.predict(inputs)
                else:
                    results  = mock_predict(inputs)

            st.success("Prediction complete!")
            st.divider()

            col_xgb, col_rf = st.columns(2)

            for col, model_key, model_label in [
                (col_xgb, "xgboost",      "XGBoost"),
                (col_rf,  "random_forest", "Random Forest"),
            ]:
                r = results[model_key]
                with col:
                    st.markdown(f"### {model_label}")
                    render_risk_badge(r["label"])
                    st.write("")
                    st.caption(f"Confidence: **{r['confidence']*100:.1f}%**")
                    st.write("**Class Probabilities**")
                    render_prob_bar(r["probabilities"])

            # Consensus
            st.divider()
            xgb_label = results["xgboost"]["label"]
            rf_label  = results["random_forest"]["label"]
            if xgb_label == rf_label:
                st.success(
                    f"✅ Both models agree: **{xgb_label} Risk**"
                )
            else:
                st.warning(
                    f"⚠️ Models disagree — XGBoost: **{xgb_label}** | "
                    f"Random Forest: **{rf_label}**. "
                    "Review individual confidence scores above."
                )

        else:
            st.info(
                "👈 Fill in the patient parameters in the sidebar "
                "and click **Predict Risk**."
            )

    # ── Tab 2: Model Comparison ───────────────────────────────────────────────
    with tab_compare:
        report_path = "artifacts/reports/model_comparison.csv"
        cm_xgb  = "artifacts/plots/xgboost_confusion_matrix.png"
        cm_rf   = "artifacts/plots/random_forest_confusion_matrix.png"
        roc_xgb = "artifacts/plots/xgboost_roc_curves.png"
        roc_rf  = "artifacts/plots/random_forest_roc_curves.png"

        if os.path.exists(report_path):
            df = pd.read_csv(report_path, index_col=0)
            st.subheader("Performance Metrics")
            st.dataframe(
                df.style.highlight_max(axis=0, color="#1a4a1a"),
                use_container_width=True,
            )
            st.caption("Green = best score for each metric")

            st.subheader("Confusion Matrices")
            c1, c2 = st.columns(2)
            if os.path.exists(cm_xgb):
                c1.image(cm_xgb, caption="XGBoost", width=500)
            if os.path.exists(cm_rf):
                c2.image(cm_rf, caption="Random Forest", width=500)

            st.subheader("ROC Curves")
            c3, c4 = st.columns(2)
            if os.path.exists(roc_xgb):
                c3.image(roc_xgb, caption="XGBoost", width=500)
            if os.path.exists(roc_rf):
                c4.image(roc_rf, caption="Random Forest", width=500)
        else:
            st.info(
                "No evaluation results yet. "
                "Run `python -m src.pipeline` to train and evaluate the models."
            )

    # ── Tab 3: About ──────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("""
### About This System

This tool predicts **cancer risk level** (Low / Medium / High) based on
demographic, behavioral, and health parameters.

**Dataset**: Cancer Patient Data (~1,000 records, 23 features)

**Models**:
- **XGBoost** — gradient boosted trees, tuned via RandomizedSearchCV
- **Random Forest** — bagged decision trees with balanced class weights

**Class Imbalance**: Handled via **SMOTEENN** (SMOTE oversampling +
Edited Nearest Neighbours cleaning) on the training set only.

**Evaluation**: 5-fold stratified cross-validation during tuning;
final metrics on a held-out test set (Accuracy, F1-weighted, ROC-AUC).

---
> ⚠️ **Disclaimer**: This tool is for educational / research purposes only
> and is **not** a substitute for professional medical diagnosis.
        """)


if __name__ == "__main__":
    main()
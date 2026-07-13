"""
pages/prediction.py
====================
Patient input form → ML prediction → probability gauge + SHAP explanation + PDF download.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from utils.preprocessor   import scale_input
from utils.explainer      import (
    get_explainer, compute_shap_values,
    plot_shap_waterfall, MODEL_TYPE_MAP,
)
from utils.report_generator import generate_pdf_report
from models.trainer import load_artifacts, get_feature_importance


# ── Feature input configuration ───────────────────────────────────────────────
FEATURE_CONFIG = {
    "age": dict(
        label="Age (years)", type="slider",
        min=20, max=100, default=50, step=1,
        help="Patient's age in years",
    ),
    "sex": dict(
        label="Sex", type="radio",
        options={0: "Female 👩", 1: "Male 👨"},
        default=1,
        help="Biological sex of the patient",
    ),
    "cp": dict(
        label="Chest Pain Type", type="select",
        options={0: "Typical Angina", 1: "Atypical Angina",
                 2: "Non-Anginal Pain", 3: "Asymptomatic"},
        default=0,
        help="Type of chest pain experienced",
    ),
    "trestbps": dict(
        label="Resting Blood Pressure (mm Hg)", type="slider",
        min=80, max=220, default=120, step=1,
        help="Resting blood pressure in mm Hg",
    ),
    "chol": dict(
        label="Serum Cholesterol (mg/dl)", type="slider",
        min=100, max=600, default=200, step=1,
        help="Serum cholesterol level in mg/dl",
    ),
    "fbs": dict(
        label="Fasting Blood Sugar > 120 mg/dl", type="radio",
        options={0: "No", 1: "Yes"},
        default=0,
        help="Whether fasting blood sugar exceeds 120 mg/dl",
    ),
    "restecg": dict(
        label="Resting ECG", type="select",
        options={0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"},
        default=0,
        help="Results of resting electrocardiogram",
    ),
    "thalach": dict(
        label="Maximum Heart Rate Achieved", type="slider",
        min=60, max=220, default=150, step=1,
        help="Maximum heart rate achieved during exercise",
    ),
    "exang": dict(
        label="Exercise-Induced Angina", type="radio",
        options={0: "No", 1: "Yes"},
        default=0,
        help="Angina triggered by exercise",
    ),
    "oldpeak": dict(
        label="ST Depression (Oldpeak)", type="number",
        min=0.0, max=10.0, default=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest",
    ),
    "slope": dict(
        label="ST Slope", type="select",
        options={0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        default=1,
        help="Slope of the peak exercise ST segment",
    ),
}


def _probability_gauge(prob: float, risk_level: str) -> go.Figure:
    """Render an animated gauge chart for disease probability."""
    color_map = {"Low": "#00D4AA", "Moderate": "#FFB347", "High": "#FF4B6E"}
    color = color_map.get(risk_level, "#6C63FF")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=42, color=color)),
        delta=dict(reference=50, valueformat=".1f",
                   increasing=dict(color="#FF4B6E"),
                   decreasing=dict(color="#00D4AA")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="white",
                      tickfont=dict(color="white")),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(30,33,48,0.8)",
            borderwidth=2,
            bordercolor="#444",
            steps=[
                dict(range=[0,   40], color="rgba(0,212,170,0.15)"),
                dict(range=[40,  65], color="rgba(255,179,71,0.15)"),
                dict(range=[65, 100], color="rgba(255,75,110,0.15)"),
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.8,
                value=prob * 100,
            ),
        ),
        title=dict(text="Heart Disease Probability", font=dict(size=16, color="white")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


def _render_input_form(feature_cols: list) -> dict:
    """Render the patient input form and return raw values dict."""
    values = {}

    with st.form("patient_form"):
        st.markdown("### 🩺 Patient Health Parameters")

        col1, col2 = st.columns(2)
        features_to_render = [f for f in feature_cols if f in FEATURE_CONFIG]

        for i, feat in enumerate(features_to_render):
            cfg = FEATURE_CONFIG[feat]
            col = col1 if i % 2 == 0 else col2

            with col:
                if cfg["type"] == "slider":
                    val = st.slider(
                        cfg["label"], cfg["min"], cfg["max"],
                        cfg["default"], cfg["step"],
                        help=cfg["help"], key=f"inp_{feat}",
                    )
                elif cfg["type"] == "number":
                    val = st.number_input(
                        cfg["label"], cfg["min"], cfg["max"],
                        cfg["default"], cfg["step"],
                        help=cfg["help"], key=f"inp_{feat}",
                    )
                elif cfg["type"] == "radio":
                    opts = list(cfg["options"].values())
                    keys = list(cfg["options"].keys())
                    sel  = st.radio(
                        cfg["label"], opts, index=keys.index(cfg["default"]),
                        horizontal=True, help=cfg["help"], key=f"inp_{feat}",
                    )
                    val = keys[opts.index(sel)]
                elif cfg["type"] == "select":
                    opts = list(cfg["options"].values())
                    keys = list(cfg["options"].keys())
                    sel  = st.selectbox(
                        cfg["label"], opts,
                        index=keys.index(cfg["default"]),
                        help=cfg["help"], key=f"inp_{feat}",
                    )
                    val = keys[opts.index(sel)]

                values[feat] = float(val)

        # For features in feature_cols but not in FEATURE_CONFIG (ca, thal), use defaults
        for feat in feature_cols:
            if feat not in values:
                values[feat] = 0.0

        submitted = st.form_submit_button(
            "🔮 Predict Heart Disease Risk",
            type="primary",
            use_container_width=True,
        )

    return values, submitted


def render_prediction():
    """Render the Prediction page."""

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔮</div>
        <h2>Heart Disease Risk Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Load artifacts ─────────────────────────────────────────────────────────
    artifacts = load_artifacts()
    if not artifacts:
        st.warning("""
        ⚠️ No trained model found. Please go to the **Model Training** page first 
        and click **Train All Models**.
        """)
        return

    model, scaler, feature_cols, model_name = artifacts

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:1rem;">
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <span style="font-size:1.5rem;">🤖</span>
            <div>
                <div style="font-weight:600; color:#E8EAED;">Active Model: {model_name}</div>
                <div style="font-size:0.8rem; color:#9AA0B4;">
                    Features: {', '.join(feature_cols)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ─────────────────────────────────────────────────────────────
    patient_values, submitted = _render_input_form(feature_cols)

    if not submitted:
        st.markdown("""
        <div style='text-align:center; padding:2rem; color:#9AA0B4;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>💡</div>
            <div>Fill in the patient parameters above and click <b>Predict</b></div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Prediction ─────────────────────────────────────────────────────────────
    input_array = np.array([[patient_values[f] for f in feature_cols]], dtype=float)
    input_scaled = scale_input(input_array[0], scaler)

    try:
        prob      = float(model.predict_proba(input_scaled)[0][1])
        pred      = int(model.predict(input_scaled)[0])
        conf      = max(prob, 1 - prob)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Risk level
    if prob < 0.35:
        risk = "Low"
        risk_class = "risk-low"
        risk_icon  = "💚"
        risk_msg   = "Low probability of heart disease. Maintain a healthy lifestyle."
    elif prob < 0.65:
        risk = "Moderate"
        risk_class = "risk-moderate"
        risk_icon  = "⚠️"
        risk_msg   = "Moderate risk detected. Consider a cardiology consultation."
    else:
        risk = "High"
        risk_class = "risk-high"
        risk_icon  = "🚨"
        risk_msg   = "High risk detected. Immediate medical evaluation is recommended."

    # ── Results Section ────────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Results")

    col_gauge, col_result = st.columns([1, 1])

    with col_gauge:
        fig_gauge = _probability_gauge(prob, risk)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_result:
        st.markdown(f"""
        <div class="prediction-card {'high' if risk=='High' else 'mod' if risk=='Moderate' else 'low'}">
            <div style="font-size:3rem; margin-bottom:0.5rem;">{risk_icon}</div>
            <div class="risk-badge {risk_class}" style="margin:0 auto 1rem; display:inline-flex;">
                {risk} Risk
            </div>
            <div style="font-size:2rem; font-weight:800; color:white; margin:0.5rem 0;">
                {prob*100:.1f}%
            </div>
            <div style="font-size:0.85rem; color:#9AA0B4; margin-bottom:1rem;">
                Disease Probability
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:0.75rem; font-size:0.85rem; color:#E8EAED; text-align:left; line-height:1.5;">
                {risk_msg}
            </div>
            <div style="margin-top:1rem; display:flex; justify-content:space-around;">
                <div style="text-align:center;">
                    <div style="font-size:1.2rem; font-weight:700; color:#6C63FF;">{conf*100:.1f}%</div>
                    <div style="font-size:0.7rem; color:#9AA0B4; text-transform:uppercase;">Confidence</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.2rem; font-weight:700; color:#6C63FF;">
                        {'Positive' if pred == 1 else 'Negative'}
                    </div>
                    <div style="font-size:0.7rem; color:#9AA0B4; text-transform:uppercase;">Diagnosis</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input Summary ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Patient Input Summary")
    input_df = pd.DataFrame([patient_values]).rename(columns={
        "age": "Age", "sex": "Sex", "cp": "Chest Pain",
        "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting BS",
        "restecg": "ECG", "thalach": "Max HR", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "slope": "ST Slope",
        "ca": "Vessels (CA)", "thal": "Thalassemia",
    }).T.reset_index()
    input_df.columns = ["Parameter", "Value"]
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    # ── SHAP Explanation ───────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🔍 SHAP Feature Contribution Explanation")
    st.markdown(
        "<div class='tooltip'>Red bars push the prediction towards heart disease; "
        "green bars push it towards healthy.</div>",
        unsafe_allow_html=True
    )

    with st.spinner("Computing SHAP explanations…"):
        try:
            shap_type = MODEL_TYPE_MAP.get(model_name, "kernel")

            # Use training data for background — load a sample
            df = load_data_safe()
            if df is not None:
                from utils.preprocessor import preprocess
                X_tr, _, _, _, _, feat_c, _ = preprocess(df)
                explainer = get_explainer(model, X_tr, shap_type)
                shap_vals  = compute_shap_values(explainer, input_scaled)

                if shap_vals is not None:
                    buf = plot_shap_waterfall(shap_vals, input_scaled, feature_cols)
                    st.image(buf, use_container_width=True)
                else:
                    st.info("SHAP explanation not available for this prediction.")
            else:
                st.info("Could not load dataset for SHAP background computation.")
        except Exception as e:
            st.warning(f"SHAP explanation skipped: {e}")

    # ── PDF Report ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📄 Download Prediction Report")

    from models.trainer import load_all_results
    saved_results = load_all_results()
    model_metrics = {}
    if model_name in saved_results:
        model_metrics = saved_results[model_name].get("metrics", {})

    try:
        pdf_bytes = generate_pdf_report(
            patient_data=patient_values,
            prediction_result={
                "probability": prob,
                "prediction":  pred,
                "risk_level":  risk,
                "confidence":  conf,
                "model_name":  model_name,
            },
            model_metrics=model_metrics,
            feature_names=feature_cols,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"heart_disease_report_{timestamp}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=False,
        )
        st.caption("🔒 Report is generated locally — no data is sent to any server.")
    except Exception as e:
        st.error(f"Could not generate PDF: {e}")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(255,75,110,0.08); border:1px solid rgba(255,75,110,0.3); 
    border-radius:12px; padding:1rem; font-size:0.85rem; color:#E8EAED; line-height:1.6;'>
        ⚕️ <b>Medical Disclaimer:</b> This AI prediction is for informational purposes only and 
        does NOT constitute medical advice. Always consult a qualified healthcare professional 
        for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)


def load_data_safe():
    """Safely load dataset without crashing if unavailable."""
    try:
        from utils.data_loader import load_data
        return load_data()
    except Exception:
        return None

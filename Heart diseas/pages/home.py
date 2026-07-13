"""
pages/home.py
=============
Dashboard landing page with hero section and animated metric cards.
"""

import streamlit as st
import plotly.graph_objects as go
from utils.data_loader import load_data


def render_metric_card(col, icon: str, value: str, label: str,
                        delta: str = "", color: str = "#6C63FF"):
    """Render a styled metric card inside a Streamlit column."""
    with col:
        st.markdown(f"""
        <div class="metric-card fade-in" style="border-left-color:{color};">
            <div style="font-size:1.8rem; margin-bottom:0.4rem;">{icon}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-label">{label}</div>
            {"<div class='metric-delta'>" + delta + "</div>" if delta else ""}
        </div>
        """, unsafe_allow_html=True)


def render_home():
    """Render the home / dashboard page."""

    # ── Hero Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner fade-in">
        <div class="hero-title">❤️ Heart Disease<br>Prediction System</div>
        <div class="hero-subtitle">
            An AI-powered healthcare analytics platform that predicts cardiovascular 
            disease risk using advanced machine learning and explainable AI.
        </div>
        <div style="margin-top:1.5rem; display:flex; gap:1rem; flex-wrap:wrap;">
            <span class="risk-badge risk-low">🧠 ML-Powered</span>
            <span class="risk-badge" style="background:#6C63FF22; color:#6C63FF; border:1px solid #6C63FF;">
                📊 Visual Analytics
            </span>
            <span class="risk-badge" style="background:#4FC3F722; color:#4FC3F7; border:1px solid #4FC3F7;">
                📄 PDF Reports
            </span>
            <span class="risk-badge" style="background:#FF6B9D22; color:#FF6B9D; border:1px solid #FF6B9D;">
                🔍 SHAP Explainability
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load dataset for stats ─────────────────────────────────────────────────
    with st.spinner("Loading dataset…"):
        try:
            df = load_data()
            n_samples    = len(df)
            n_features   = len(df.columns) - 1
            disease_pct  = int(df["target"].mean() * 100)
            healthy_pct  = 100 - disease_pct
        except Exception:
            n_samples   = 303
            n_features  = 13
            disease_pct = 54
            healthy_pct = 46

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📈 Dataset at a Glance")

    c1, c2, c3, c4 = st.columns(4)
    render_metric_card(c1, "🗃️", str(n_samples), "Total Patients",
                       "UCI Cleveland Dataset", "#6C63FF")
    render_metric_card(c2, "🔬", str(n_features), "Clinical Features",
                       "Input parameters", "#00D4AA")
    render_metric_card(c3, "❤️‍🩹", f"{disease_pct}%", "Disease Prevalence",
                       "Positive cases", "#FF4B6E")
    render_metric_card(c4, "💚", f"{healthy_pct}%", "Healthy Patients",
                       "Negative cases", "#FFB347")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Feature mini-donut ─────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 🩺 Class Distribution")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Heart Disease", "Healthy"],
            values=[disease_pct, healthy_pct],
            hole=0.65,
            marker=dict(colors=["#FF4B6E", "#00D4AA"],
                        line=dict(color="#0E1117", width=3)),
            textinfo="label+percent",
            textfont=dict(color="white", size=13),
        )])
        fig_pie.add_annotation(
            text=f"<b>{n_samples}</b><br>Patients",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="white"),
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(bgcolor="rgba(30,33,48,0.8)", bordercolor="#444",
                        borderwidth=1, orientation="h", y=-0.1),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("### 🤖 ML Models Available")
        models = [
            ("🔵", "Logistic Regression",   "Fast baseline"),
            ("🟣", "Random Forest",          "Best for complex patterns"),
            ("🟡", "Decision Tree",          "Interpretable"),
            ("🟠", "K-Nearest Neighbors",    "Non-parametric"),
            ("🔴", "Support Vector Machine", "High-dimensional"),
            ("🟢", "XGBoost",               "Gradient boosting"),
        ]
        for icon, name, desc in models:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.5rem; padding:0.75rem 1rem;">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <div>
                        <div style="font-weight:600; color:#E8EAED; font-size:0.9rem;">{name}</div>
                        <div style="font-size:0.75rem; color:#9AA0B4;">{desc}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Navigation Cards ───────────────────────────────────────────────────────
    st.markdown("### 🚀 Quick Navigation")
    nav_cols = st.columns(4)
    nav_items = [
        ("📊", "Data Explorer",   "Explore dataset patterns, distributions & correlations",  "#6C63FF"),
        ("🧪", "Model Training",  "Train & compare 6 ML models side-by-side",               "#00D4AA"),
        ("🔮", "Prediction",      "Input patient data and get instant risk assessment",       "#FF4B6E"),
        ("ℹ️",  "About",          "Learn about the dataset, model, and methodology",          "#FFB347"),
    ]
    for col, (icon, title, desc, color) in zip(nav_cols, nav_items):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}; cursor:pointer; min-height:130px;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700; color:#E8EAED; font-size:1rem; margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.8rem; color:#9AA0B4; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#5a6075; font-size:0.8rem;'>"
        "⚕️ <b>Disclaimer:</b> This tool is for educational and research purposes only. "
        "Always consult a qualified medical professional for clinical decisions."
        "</div>",
        unsafe_allow_html=True
    )

"""
app.py
======
Heart Disease Prediction System — Main Streamlit Application

Entry point: streamlit run app.py
"""

import os
import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com",
        "Report a Bug": "https://github.com",
        "About":        "AI-Powered Heart Disease Prediction System",
    }
)

# ── Load custom CSS ────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# ── Additional inline overrides for dark theme ─────────────────────────────────
st.markdown("""
<style>
/* Force dark background across all Streamlit internals */
.stApp {
    background-color: #0E1117 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #1E2130 !important;
}
/* Dataframe styling */
.stDataFrame { border-radius: 12px; overflow: hidden; }
/* Expander */
.streamlit-expanderHeader {
    background: #1E2130 !important;
    border-radius: 8px !important;
    color: #E8EAED !important;
}
/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6C63FF, #00D4AA) !important;
    border-radius: 999px !important;
}
/* Spinner */
.stSpinner > div { border-top-color: #6C63FF !important; }
/* Radio */
.stRadio > div { gap: 0.5rem; }
/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    padding: 0.75rem !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
}
/* Download button */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #00D4AA, #6C63FF) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
}
/* Info / Success / Warning boxes */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Import page renderers ──────────────────────────────────────────────────────
from pages.home             import render_home
from pages.data_exploration import render_data_exploration
from pages.model_training   import render_model_training
from pages.prediction       import render_prediction
from pages.about            import render_about

# ── Navigation definition ──────────────────────────────────────────────────────
PAGES = {
    "🏠 Home":             render_home,
    "📊 Data Explorer":    render_data_exploration,
    "🧪 Model Training":   render_model_training,
    "🔮 Prediction":       render_prediction,
    "ℹ️ About":            render_about,
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem; margin-bottom:0.25rem;">❤️</div>
        <div style="font-weight:800; font-size:1.1rem; color:#E8EAED; line-height:1.2;">
            Heart Disease<br>Prediction System
        </div>
        <div style="font-size:0.75rem; color:#9AA0B4; margin-top:0.3rem;">
            AI-Powered Healthcare Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a2d3e; margin:0.75rem 0;'>", unsafe_allow_html=True)

    # Navigation
    st.markdown("#### 🧭 Navigation")
    selected_page = st.radio(
        "Navigate to:",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown("<hr style='border-color:#2a2d3e; margin:0.75rem 0;'>", unsafe_allow_html=True)

    # System status
    st.markdown("#### 📡 System Status")
    import os
    models_dir = os.path.join(os.path.dirname(__file__), "models", "saved")
    model_trained = os.path.exists(os.path.join(models_dir, "best_model.pkl"))

    if model_trained:
        from models.trainer import load_artifacts
        arts = load_artifacts()
        if arts:
            _, _, _, best_name = arts
            st.markdown(f"""
            <div style="background:#00D4AA15; border:1px solid #00D4AA33; border-radius:8px; padding:0.6rem 0.8rem; font-size:0.8rem;">
                ✅ <b>Model Ready</b><br>
                <span style="color:#9AA0B4;">{best_name}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FF4B6E15; border:1px solid #FF4B6E33; border-radius:8px; padding:0.6rem 0.8rem; font-size:0.8rem;">
            ⚠️ <b>No Model Trained</b><br>
            <span style="color:#9AA0B4;">Go to Model Training</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a2d3e; margin:0.75rem 0;'>", unsafe_allow_html=True)

    # Dataset status
    data_path = os.path.join(os.path.dirname(__file__), "data", "heart.csv")
    if os.path.exists(data_path):
        st.markdown("""
        <div style="font-size:0.75rem; color:#00D4AA;">
            📂 Dataset: Loaded ✅
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.75rem; color:#FFB347;">
            📂 Dataset: Will auto-download ⬇️
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position:absolute; bottom:1.5rem; left:0; right:0; text-align:center; 
    font-size:0.7rem; color:#5a6075; padding:0 1rem;">
        v1.0.0 &nbsp;•&nbsp; UCI Heart Disease Dataset<br>
        ⚕️ For research purposes only
    </div>
    """, unsafe_allow_html=True)

# ── Render selected page ───────────────────────────────────────────────────────
PAGES[selected_page]()

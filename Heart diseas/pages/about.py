"""
pages/about.py
==============
About page: project info, tech stack, dataset citation, methodology.
"""

import streamlit as st


def render_about():
    """Render the About page."""

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">ℹ️</div>
        <h2>About This Project</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title" style="font-size:2rem;">
            ❤️ AI-Powered Heart Disease Prediction
        </div>
        <div class="hero-subtitle">
            A production-ready machine learning system for cardiovascular risk assessment, 
            built with explainable AI and modern healthcare analytics best practices.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗃️ Dataset", "⚙️ Tech Stack", "🔬 Methodology", "📜 Citation"]
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🗃️ UCI Heart Disease Dataset (Cleveland)")
        col_a, col_b = st.columns([1.6, 1])
        with col_a:
            st.markdown("""
            The **Cleveland Heart Disease Database** is one of the most widely used 
            benchmarks in medical machine learning research. It contains 303 patient 
            records collected at the Cleveland Clinic Foundation.

            #### Key Facts
            | Property | Value |
            |---|---|
            | Source | UCI Machine Learning Repository |
            | Instances | 303 patients |
            | Features | 13 clinical attributes + 1 target |
            | Task | Binary classification |
            | Missing Values | Present (ca, thal) |
            | License | CC BY 4.0 |

            #### Features Used
            | Feature | Type | Description |
            |---|---|---|
            | Age | Continuous | Patient age in years |
            | Sex | Binary | 1=Male, 0=Female |
            | Chest Pain (CP) | Ordinal | 0-3 scale |
            | Resting BP | Continuous | mm Hg |
            | Cholesterol | Continuous | mg/dl |
            | Fasting Blood Sugar | Binary | >120 mg/dl |
            | Resting ECG | Ordinal | 0=Normal, 1=ST-T, 2=LVH |
            | Max Heart Rate | Continuous | Beats per minute |
            | Exercise Angina | Binary | Yes/No |
            | ST Depression | Continuous | Oldpeak value |
            | ST Slope | Ordinal | 0=Up, 1=Flat, 2=Down |
            | CA | Ordinal | Major vessels (0-3) |
            | Thal | Ordinal | Thalassemia type |
            """)
        with col_b:
            # Stats cards
            stats = [
                ("303", "Patients",        "#6C63FF"),
                ("13",  "Features",        "#00D4AA"),
                ("54%", "Disease Rate",    "#FF4B6E"),
                ("46%", "Healthy Rate",    "#FFB347"),
                ("6",   "ML Models",       "#4FC3F7"),
                ("5+",  "Visualizations",  "#FF6B9D"),
            ]
            for val, label, color in stats:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:0.6rem; padding:0.75rem 1rem;">
                    <div class="metric-value" style="color:{color}; font-size:1.4rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### ⚙️ Technology Stack")

        tech_items = [
            ("🐍", "Python 3.10+",    "Core language",                            "#6C63FF"),
            ("🌊", "Streamlit",       "Web application framework",                "#FF4B6E"),
            ("🐼", "Pandas",          "Data manipulation & analysis",              "#00D4AA"),
            ("🔢", "NumPy",           "Numerical computing",                       "#FFB347"),
            ("🤖", "Scikit-learn",    "Machine learning models & preprocessing",   "#4FC3F7"),
            ("📈", "Plotly",          "Interactive charts & visualizations",        "#6C63FF"),
            ("🔍", "SHAP",            "Model explainability (SHapley values)",      "#FF6B9D"),
            ("📄", "ReportLab",       "PDF report generation",                      "#00D4AA"),
            ("💾", "Joblib",          "Model serialization & caching",              "#FFB347"),
            ("🌳", "XGBoost",         "Gradient boosted trees (optional)",          "#4FC3F7"),
            ("📊", "Matplotlib",      "Static chart rendering for SHAP",            "#FF4B6E"),
            ("🎨", "Seaborn",         "Statistical visualizations",                 "#6C63FF"),
        ]

        cols = st.columns(3)
        for i, (icon, name, desc, color) in enumerate(tech_items):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:0.6rem; padding:0.9rem 1rem;">
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <span style="font-size:1.4rem;">{icon}</span>
                        <div>
                            <div style="font-weight:600; color:{color}; font-size:0.9rem;">{name}</div>
                            <div style="font-size:0.75rem; color:#9AA0B4;">{desc}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Methodology ────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🔬 ML Methodology")
        st.markdown("""
        #### 1. Data Preprocessing
        - **Missing value imputation**: Median for continuous, mode for categorical
        - **Duplicate removal**: Ensures data quality
        - **Feature scaling**: StandardScaler (zero mean, unit variance)
        - **Adaptive feature selection**: Drops features with >20% missingness
        - **Stratified split**: Maintains class ratio in train/test sets

        #### 2. Model Training
        Six classifiers are trained and evaluated using 5-fold cross-validation awareness:
        
        | Model | Key Strength |
        |---|---|
        | Logistic Regression | Fast, interpretable baseline |
        | Random Forest (200 trees) | Handles non-linearity, robust to overfitting |
        | Decision Tree | Human-interpretable rules |
        | K-Nearest Neighbors | Non-parametric, no distribution assumption |
        | Support Vector Machine | Effective in high dimensions |
        | XGBoost | State-of-the-art gradient boosting |

        #### 3. Model Selection
        Best model is automatically selected by **ROC-AUC** score on the held-out test set.

        #### 4. Explainability (SHAP)
        - **TreeExplainer**: Used for Random Forest, Decision Tree, XGBoost
        - **LinearExplainer**: Used for Logistic Regression
        - **KernelExplainer**: Universal fallback for SVM, KNN
        - Output: Waterfall plot shows per-feature contribution magnitude and direction

        #### 5. Risk Stratification
        | Probability | Risk Level | Recommendation |
        |---|---|---|
        | 0% – 35% | 🟢 Low | Maintain healthy lifestyle |
        | 35% – 65% | 🟡 Moderate | Consult cardiologist |
        | 65% – 100% | 🔴 High | Immediate medical evaluation |
        """)

    # ── Citation ────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📜 Dataset Citation")
        st.code("""
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).
Heart Disease [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
        """, language="text")

        st.markdown("### 📚 Key References")
        references = [
            ("Detrano et al. (1989)",
             "International application of a new probability algorithm for the diagnosis of coronary artery disease.",
             "American Journal of Cardiology, 64(5), 304-310."),
            ("SHAP (Lundberg & Lee, 2017)",
             "A Unified Approach to Interpreting Model Predictions.",
             "NeurIPS 2017."),
            ("Chen & Guestrin (2016)",
             "XGBoost: A Scalable Tree Boosting System.",
             "KDD 2016."),
        ]
        for title, desc, source in references:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.75rem; padding:0.85rem 1rem;">
                <div style="font-weight:600; color:#6C63FF; margin-bottom:0.25rem;">{title}</div>
                <div style="font-size:0.85rem; color:#E8EAED; margin-bottom:0.2rem;">{desc}</div>
                <div style="font-size:0.75rem; color:#9AA0B4;"><i>{source}</i></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='text-align:center; color:#5a6075; font-size:0.8rem; padding:1rem;'>
            Built with ❤️ as a production-ready ML healthcare application.<br>
            For educational and research purposes only.
        </div>
        """, unsafe_allow_html=True)

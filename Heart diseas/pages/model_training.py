"""
pages/model_training.py
========================
Train & compare ML models, display leaderboard, confusion matrix,
ROC curves, and feature importance.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader   import load_data
from utils.preprocessor  import preprocess
from utils.metrics       import (
    plot_confusion_matrix, plot_roc_curves,
    plot_metrics_comparison, plot_feature_importance,
)
from models.trainer import (
    train_all_models, save_artifacts, load_artifacts,
    load_all_results, get_feature_importance,
)


def render_model_training():
    """Render the Model Training & Evaluation page."""

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧪</div>
        <h2>Model Training & Evaluation</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Training Config")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
        random_state = st.number_input("Random Seed", 0, 999, 42)
        st.markdown("---")

    # ── Status check ───────────────────────────────────────────────────────────
    artifacts = load_artifacts()
    saved_results = load_all_results()

    if artifacts:
        model, scaler, feature_cols, model_name = artifacts
        st.success(f"✅ Trained models found! Best model: **{model_name}**")
    else:
        st.info("💡 No saved models found. Click **Train Models** to begin.")
        model = scaler = feature_cols = model_name = None

    # ── Train button ───────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        train_clicked = st.button("🚀 Train All Models", type="primary",
                                  use_container_width=True)

    if train_clicked:
        with st.spinner("Loading and preprocessing data…"):
            df = load_data()
            X_train, X_test, y_train, y_test, scaler, feature_cols, df_clean = \
                preprocess(df, test_size=test_size, random_state=int(random_state))

        # Progress bar
        progress_bar  = st.progress(0, text="Starting training…")
        status_text   = st.empty()

        def progress_cb(frac, msg):
            progress_bar.progress(frac, text=msg)
            status_text.markdown(f"⏳ {msg}")

        all_results, best_name, best_model = train_all_models(
            X_train, X_test, y_train, y_test,
            random_state=int(random_state),
            progress_callback=progress_cb,
        )

        save_artifacts(best_model, scaler, feature_cols, best_name, all_results)
        progress_bar.progress(1.0, "✅ All models trained and saved!")
        status_text.empty()

        model      = best_model
        model_name = best_name
        saved_results = load_all_results()

        # Store in session state for this page
        st.session_state["_train_results"] = all_results
        st.session_state["_train_X_test"]  = X_test
        st.session_state["_train_y_test"]  = y_test
        st.session_state["_feature_cols"]  = feature_cols

        st.balloons()
        st.success(f"🏆 Best model selected: **{best_name}**")

    # ── If we have results (from this run or session) ─────────────────────────
    all_results = st.session_state.get("_train_results", None)
    X_test      = st.session_state.get("_train_X_test", None)
    y_test_ses  = st.session_state.get("_train_y_test", None)
    feature_cols_ses = st.session_state.get("_feature_cols", feature_cols)

    if not saved_results and not all_results:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#9AA0B4;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🤖</div>
            <div style='font-size:1.2rem;'>Train models first to see evaluation results</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Use live results if available, else load lite version
    results_display = all_results if all_results else saved_results

    # ── Leaderboard ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🏆 Model Leaderboard")

    leaderboard_rows = []
    for name, data in results_display.items():
        m = data["metrics"]
        row = {
            "Model":     name,
            "Accuracy":  m.get("Accuracy",  0),
            "Precision": m.get("Precision", 0),
            "Recall":    m.get("Recall",    0),
            "F1 Score":  m.get("F1 Score",  0),
            "ROC-AUC":   m.get("ROC-AUC",   0) or 0,
        }
        leaderboard_rows.append(row)

    lb_df = pd.DataFrame(leaderboard_rows).sort_values("ROC-AUC", ascending=False)
    lb_df = lb_df.reset_index(drop=True)
    lb_df.index += 1  # rank from 1

    # Add medal emoji for top 3
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lb_df.insert(0, "Rank", [medals.get(i, f"#{i}") for i in lb_df.index])

    st.dataframe(
        lb_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall":   "{:.4f}", "F1 Score":  "{:.4f}",
            "ROC-AUC":  "{:.4f}",
        }).background_gradient(subset=["ROC-AUC"], cmap="Blues"),
        use_container_width=True, hide_index=True
    )

    # ── Metrics Comparison Chart ────────────────────────────────────────────────
    metrics_only = {k: v["metrics"] for k, v in results_display.items()}
    st.plotly_chart(plot_metrics_comparison(metrics_only), use_container_width=True)

    # ── ROC Curves ─────────────────────────────────────────────────────────────
    st.markdown("### 📈 ROC Curves — All Models")
    roc_data = {k: v["roc"] for k, v in results_display.items()
                if "roc" in v and v["roc"]}
    if roc_data:
        st.plotly_chart(plot_roc_curves(roc_data), use_container_width=True)

    # ── Confusion Matrix + Feature Importance (best model) ────────────────────
    if artifacts:
        best_model_loaded, _, feat_cols, best_model_name = artifacts
    else:
        best_model_loaded = None
        best_model_name   = model_name
        feat_cols         = feature_cols_ses

    st.markdown("### 🔬 Best Model Deep Dive")
    col_cm, col_fi = st.columns(2)

    with col_cm:
        if all_results and best_model_name in all_results:
            y_pred = all_results[best_model_name]["y_pred"]
            fig_cm = plot_confusion_matrix(y_test_ses, y_pred, best_model_name)
            st.plotly_chart(fig_cm, use_container_width=True)
        elif best_model_name in results_display:
            st.info("Retrain to view confusion matrix.")

    with col_fi:
        if best_model_loaded and feat_cols:
            importances = get_feature_importance(best_model_loaded, feat_cols)
            if importances is not None:
                fig_fi = plot_feature_importance(
                    importances, feat_cols, best_model_name
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

    # ── Detailed Metrics Table ─────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Metrics by Model")
    for name, data in results_display.items():
        m = data["metrics"]
        is_best = (name == best_model_name)
        prefix = "🏆 " if is_best else ""
        with st.expander(f"{prefix}{name}", expanded=is_best):
            metric_cols = st.columns(5)
            metric_items = [
                ("🎯 Accuracy",  m.get("Accuracy",  0),  "#6C63FF"),
                ("🔍 Precision", m.get("Precision", 0),  "#00D4AA"),
                ("📡 Recall",    m.get("Recall",    0),  "#FF4B6E"),
                ("⚖️ F1 Score",  m.get("F1 Score",  0),  "#FFB347"),
                ("📈 ROC-AUC",   m.get("ROC-AUC",   0) or 0, "#4FC3F7"),
            ]
            for mc, (lbl, val, color) in zip(metric_cols, metric_items):
                with mc:
                    st.markdown(f"""
                    <div class="metric-card" style="padding:0.75rem; text-align:center;">
                        <div class="metric-value" style="color:{color}; font-size:1.4rem;">
                            {val:.4f}
                        </div>
                        <div class="metric-label" style="font-size:0.7rem;">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

"""
explainer.py
============
SHAP-based model explainability.
Supports TreeExplainer (RF, DT, XGB) and LinearExplainer (LR, SVM).
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO


def get_explainer(model, X_train: np.ndarray, model_type: str = "tree"):
    """
    Create an appropriate SHAP explainer based on model type.

    Parameters
    ----------
    model       : trained sklearn model
    X_train     : training data (numpy array)
    model_type  : "tree" | "linear" | "kernel"
    """
    try:
        if model_type == "tree":
            return shap.TreeExplainer(model)
        elif model_type == "linear":
            return shap.LinearExplainer(model, X_train,
                                        feature_perturbation="interventional")
        else:
            # KernelExplainer works for any model — slow but universal
            background = shap.kmeans(X_train, 50)
            return shap.KernelExplainer(model.predict_proba, background)
    except Exception:
        # Universal fallback
        background = shap.kmeans(X_train, min(50, len(X_train)))
        return shap.KernelExplainer(model.predict_proba, background)


def compute_shap_values(explainer, X_input: np.ndarray):
    """
    Compute SHAP values for a single input or batch.
    Returns raw shap_values (array or Explanation object).
    """
    try:
        shap_values = explainer.shap_values(X_input)
        return shap_values
    except Exception as e:
        st.warning(f"SHAP computation warning: {e}")
        return None


def plot_shap_waterfall(shap_values, X_input: np.ndarray,
                         feature_names: list, class_idx: int = 1) -> BytesIO:
    """
    Generate a SHAP waterfall plot for a single prediction.
    Returns a BytesIO PNG image.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    try:
        # Handle both old (list) and new (Explanation) SHAP API
        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]
            base = getattr(explainer, "expected_value", [0, 0])
            base_val = base[class_idx] if isinstance(base, (list, np.ndarray)) else base
        else:
            sv = shap_values[0]
            base_val = 0.5  # fallback

        # Sort by absolute magnitude
        idx_sorted = np.argsort(np.abs(sv))[::-1][:10]
        names_sorted = [feature_names[i] for i in idx_sorted]
        vals_sorted  = sv[idx_sorted]

        colors = ["#FF4B6E" if v > 0 else "#00D4AA" for v in vals_sorted]

        bars = ax.barh(range(len(vals_sorted)), vals_sorted, color=colors, alpha=0.85)
        ax.set_yticks(range(len(vals_sorted)))
        ax.set_yticklabels(names_sorted, color="white", fontsize=11)
        ax.set_xlabel("SHAP Value (Impact on Prediction)", color="white", fontsize=11)
        ax.set_title("Feature Contribution to Prediction", color="white", fontsize=14,
                     fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(x=0, color="#666", linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, vals_sorted):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                color="white", fontsize=9,
            )

        plt.tight_layout()

    except Exception:
        ax.text(0.5, 0.5, "SHAP visualization unavailable",
                ha="center", va="center", color="white", transform=ax.transAxes)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0E1117")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_shap_summary(shap_values, X_data: np.ndarray,
                       feature_names: list, class_idx: int = 1) -> BytesIO:
    """
    Generate a SHAP summary (beeswarm) plot for global feature importance.
    Returns a BytesIO PNG image.
    """
    plt.style.use("dark_background")

    try:
        if isinstance(shap_values, list):
            sv = shap_values[class_idx]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        shap.summary_plot(
            sv, X_data,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            color="#6C63FF",
        )
        plt.title("Global Feature Importance (SHAP)", color="white",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#0E1117")
        plt.close("all")
        buf.seek(0)
        return buf

    except Exception:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0E1117")
        ax.text(0.5, 0.5, "SHAP summary unavailable",
                ha="center", va="center", color="white", transform=ax.transAxes)
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor="#0E1117")
        plt.close(fig)
        buf.seek(0)
        return buf


# Map model class names to SHAP explainer types
MODEL_TYPE_MAP = {
    "Logistic Regression":  "linear",
    "Random Forest":        "tree",
    "Decision Tree":        "tree",
    "K-Nearest Neighbors":  "kernel",
    "Support Vector Machine": "kernel",
    "XGBoost":              "tree",
}

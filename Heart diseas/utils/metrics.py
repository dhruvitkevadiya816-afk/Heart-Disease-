"""
metrics.py
==========
Evaluation metrics and Plotly visualizations for model comparison.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)


# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#6C63FF",
    "success":   "#00D4AA",
    "danger":    "#FF4B6E",
    "warning":   "#FFB347",
    "info":      "#4FC3F7",
    "bg_dark":   "#0E1117",
    "card_dark": "#1E2130",
}

MODEL_COLORS = [
    "#6C63FF", "#00D4AA", "#FF4B6E",
    "#FFB347", "#4FC3F7", "#FF6B9D",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """Compute all classification metrics."""
    metrics = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["ROC-AUC"] = None
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "Model") -> go.Figure:
    """Render an annotated confusion matrix as a Plotly heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Disease", "Heart Disease"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#1E2130"], [0.5, "#6C63FF"], [1, "#00D4AA"]],
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 22, "color": "white"},
    ))

    # Annotate with TP/TN/FP/FN labels
    annotations = [
        dict(x="No Disease",    y="No Disease",    text="TN", showarrow=False,
             font=dict(color="#aaa", size=12), xshift=30, yshift=-15),
        dict(x="Heart Disease", y="No Disease",    text="FP", showarrow=False,
             font=dict(color="#aaa", size=12), xshift=30, yshift=-15),
        dict(x="No Disease",    y="Heart Disease", text="FN", showarrow=False,
             font=dict(color="#aaa", size=12), xshift=30, yshift=-15),
        dict(x="Heart Disease", y="Heart Disease", text="TP", showarrow=False,
             font=dict(color="#aaa", size=12), xshift=30, yshift=-15),
    ]

    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=16)),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_roc_curves(roc_data: dict) -> go.Figure:
    """
    Plot ROC curves for multiple models on one chart.
    roc_data: {model_name: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#555", width=1),
        name="Random Classifier",
        showlegend=True,
    ))

    for i, (name, data) in enumerate(roc_data.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        fig.add_trace(go.Scatter(
            x=data["fpr"],
            y=data["tpr"],
            mode="lines",
            name=f"{name} (AUC={data['auc']:.3f})",
            line=dict(color=color, width=2.5),
            fill="tozeroy" if i == 0 else None,
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.05)",
        ))

    fig.update_layout(
        title=dict(text="ROC Curves — All Models", font=dict(size=16)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(30,33,48,0.8)", bordercolor="#444", borderwidth=1),
        xaxis=dict(gridcolor="#2a2a3a", zeroline=False),
        yaxis=dict(gridcolor="#2a2a3a", zeroline=False),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_metrics_comparison(results: dict) -> go.Figure:
    """
    Grouped bar chart comparing metrics across models.
    results: {model_name: {metric_name: value}}
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    model_names  = list(results.keys())

    fig = go.Figure()
    for i, model in enumerate(model_names):
        vals = [results[model].get(m, 0) or 0 for m in metric_names]
        fig.add_trace(go.Bar(
            name=model,
            x=metric_names,
            y=vals,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Model Performance Comparison", font=dict(size=16)),
        yaxis=dict(range=[0, 1.15], gridcolor="#2a2a3a", zeroline=False),
        xaxis=dict(gridcolor="#2a2a3a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(30,33,48,0.8)", bordercolor="#444", borderwidth=1),
        height=420,
        margin=dict(l=20, r=20, t=50, b=60),
    )
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                             model_name: str = "Model") -> go.Figure:
    """Horizontal bar chart of feature importances."""
    sorted_idx = np.argsort(importances)
    colors_grad = px.colors.sequential.Viridis[::-1]

    fig = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation="h",
        marker=dict(
            color=importances[sorted_idx],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
    ))

    fig.update_layout(
        title=dict(text=f"Feature Importance — {model_name}", font=dict(size=16)),
        xaxis_title="Importance Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2a2a3a", zeroline=False),
        yaxis=dict(gridcolor="#2a2a3a"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def get_roc_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute FPR, TPR, and AUC for a single model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}

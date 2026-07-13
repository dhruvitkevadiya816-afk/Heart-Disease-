"""
trainer.py
==========
Trains, evaluates, and persists multiple ML models.
Automatically selects the best-performing model by ROC-AUC.
"""

import os
import numpy as np
import joblib
import streamlit as st

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.svm           import SVC

from utils.metrics import compute_metrics, get_roc_data

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved")


def get_model_definitions(random_state: int = 42) -> dict:
    """Return a dict of {name: (estimator, shap_type)} tuples."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=random_state, C=1.0),
            "linear",
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=None,
                                   random_state=random_state, n_jobs=-1),
            "tree",
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state, max_depth=8),
            "tree",
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_neighbors=7, metric="minkowski"),
            "kernel",
        ),
        "Support Vector Machine": (
            SVC(probability=True, kernel="rbf", C=1.0, gamma="scale",
                random_state=random_state),
            "kernel",
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(n_estimators=200, learning_rate=0.05,
                          max_depth=6, use_label_encoder=False,
                          eval_metric="logloss", random_state=random_state,
                          verbosity=0),
            "tree",
        )
    return models


def train_all_models(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    random_state: int = 42,
    progress_callback=None,
) -> tuple:
    """
    Train all models, evaluate, and return results.

    Returns
    -------
    results     : {model_name: {"metrics": dict, "roc": dict, "model": estimator,
                                "shap_type": str, "y_pred": array, "y_prob": array}}
    best_name   : str — name of the best model by ROC-AUC
    best_model  : estimator
    """
    model_defs = get_model_definitions(random_state)
    results    = {}
    n_models   = len(model_defs)

    for i, (name, (estimator, shap_type)) in enumerate(model_defs.items()):
        if progress_callback:
            progress_callback(i / n_models, f"Training {name}…")

        try:
            # Train
            estimator.fit(X_train, y_train)

            # Predict
            y_pred = estimator.predict(X_test)
            y_prob = estimator.predict_proba(X_test)[:, 1]

            # Metrics
            metrics  = compute_metrics(y_test, y_pred, y_prob)
            roc_data = get_roc_data(y_test, y_prob)

            results[name] = {
                "model":     estimator,
                "shap_type": shap_type,
                "metrics":   metrics,
                "roc":       roc_data,
                "y_pred":    y_pred,
                "y_prob":    y_prob,
            }
        except Exception as e:
            st.warning(f"Could not train {name}: {e}")

    if progress_callback:
        progress_callback(1.0, "All models trained!")

    # ── Select best model by ROC-AUC ──────────────────────────────────────────
    best_name = max(
        results.keys(),
        key=lambda k: results[k]["metrics"].get("ROC-AUC") or 0
    )
    best_model = results[best_name]["model"]

    return results, best_name, best_model


def save_artifacts(
    model, scaler, feature_cols: list, model_name: str,
    all_results: dict = None
):
    """Persist model, scaler, feature list, and results to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,        os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(model_name,   os.path.join(MODELS_DIR, "best_model_name.pkl"))

    if all_results:
        # Save lightweight version (no raw arrays for storage efficiency)
        lite = {
            k: {"metrics": v["metrics"], "roc": v["roc"], "shap_type": v["shap_type"]}
            for k, v in all_results.items()
        }
        joblib.dump(lite, os.path.join(MODELS_DIR, "all_results.pkl"))


def load_artifacts() -> tuple:
    """
    Load persisted model, scaler, and feature list.
    Returns (model, scaler, feature_cols, model_name) or None if not found.
    """
    try:
        model        = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        model_name   = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
        return model, scaler, feature_cols, model_name
    except FileNotFoundError:
        return None


def load_all_results() -> dict:
    """Load the lightweight results dict if available."""
    path = os.path.join(MODELS_DIR, "all_results.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


def get_feature_importance(model, feature_names: list) -> np.ndarray:
    """
    Extract feature importances from a model.
    Returns None if the model doesn't support it.
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return np.abs(coef)
    return None

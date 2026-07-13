"""
data_loader.py
==============
Handles downloading and loading the UCI Heart Disease dataset.
Uses Streamlit caching for performance.
"""

import os
import io
import requests
import pandas as pd
import streamlit as st

# ── Column names as defined by UCI Cleveland dataset ──────────────────────────
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

# Human-readable display names
DISPLAY_NAMES = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Cholesterol",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "ST Slope",
    "ca":       "Major Vessels (CA)",
    "thal":     "Thalassemia",
    "target":   "Heart Disease",
}

# Primary download URL (UCI Cleveland dataset via GitHub mirror)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FALLBACK_URL = (
    "https://raw.githubusercontent.com/rashida048/Datasets/master/"
    "heart.csv"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "heart.csv")


def _download_uci() -> pd.DataFrame:
    """Download from the UCI repository (Cleveland format with '?' for missing)."""
    resp = requests.get(DATASET_URL, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        header=None,
        names=UCI_COLUMNS,
        na_values="?"
    )
    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)
    return df


def _download_fallback() -> pd.DataFrame:
    """Download a pre-cleaned version as fallback."""
    resp = requests.get(FALLBACK_URL, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    # Rename 'condition' → 'target' if present
    if "condition" in df.columns:
        df.rename(columns={"condition": "target"}, inplace=True)
    return df


def download_dataset() -> pd.DataFrame:
    """
    Download the UCI Heart Disease dataset and cache it locally.
    Tries the official UCI URL first, then a GitHub mirror.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    try:
        df = _download_uci()
    except Exception:
        try:
            df = _download_fallback()
        except Exception as e:
            raise RuntimeError(
                f"Could not download the dataset from any source: {e}"
            )

    df.to_csv(DATA_PATH, index=False)
    return df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load dataset with Streamlit caching.
    Downloads if not already on disk.
    """
    return download_dataset()


def get_feature_descriptions() -> dict:
    """Return human-readable descriptions for each feature."""
    return {
        "age":      "Age of the patient in years",
        "sex":      "Sex (1 = Male, 0 = Female)",
        "cp":       "Chest pain type (0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)",
        "trestbps": "Resting blood pressure in mm Hg",
        "chol":     "Serum cholesterol in mg/dl",
        "fbs":      "Fasting blood sugar > 120 mg/dl (1=True, 0=False)",
        "restecg":  "Resting ECG results (0=Normal, 1=ST-T Abnormality, 2=LVH)",
        "thalach":  "Maximum heart rate achieved",
        "exang":    "Exercise-induced angina (1=Yes, 0=No)",
        "oldpeak":  "ST depression induced by exercise relative to rest",
        "slope":    "Slope of peak exercise ST segment (0=Up, 1=Flat, 2=Down)",
        "ca":       "Number of major vessels colored by fluoroscopy (0–3)",
        "thal":     "Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)",
    }

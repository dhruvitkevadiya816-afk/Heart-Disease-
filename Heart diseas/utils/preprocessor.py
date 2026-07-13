"""
preprocessor.py
===============
Full preprocessing pipeline:
  - Missing value imputation
  - Duplicate removal
  - Label encoding
  - Standard scaling
  - Train/test split
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st


# Features used for model training (drop 'ca' and 'thal' if >20% missing)
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope"
]
TARGET_COL = "target"

# Extended set including ca / thal (used when data is clean enough)
EXTENDED_FEATURE_COLS = FEATURE_COLS + ["ca", "thal"]

# Categorical features (ordinal-encoded)
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope"]


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    scaler                            : fitted StandardScaler
    feature_cols                      : list of feature names used
    df_clean                          : cleaned DataFrame (before scaling)
    """
    df = df.copy()

    # ── 1. Remove duplicates ──────────────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── 2. Binarise target (already done in loader, but safety check) ─────────
    if df[TARGET_COL].nunique() > 2:
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    # ── 3. Select feature set based on missingness ────────────────────────────
    missing_pct = df.isnull().mean()
    feature_cols = EXTENDED_FEATURE_COLS.copy()
    for col in ["ca", "thal"]:
        if col in df.columns and missing_pct.get(col, 0) > 0.20:
            feature_cols.remove(col)
        elif col not in df.columns:
            if col in feature_cols:
                feature_cols.remove(col)

    # Keep only relevant columns
    keep_cols = [c for c in feature_cols if c in df.columns] + [TARGET_COL]
    df = df[keep_cols].copy()
    feature_cols = [c for c in feature_cols if c in df.columns]

    # ── 4. Impute missing values ──────────────────────────────────────────────
    for col in df.columns:
        if df[col].isnull().any():
            if col in CATEGORICAL_COLS:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # ── 5. Convert to numeric (safety) ────────────────────────────────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    df_clean = df.copy()

    # ── 6. Split features / target ────────────────────────────────────────────
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # ── 7. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 8. Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols, df_clean


def scale_input(raw_input: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale a single patient input vector using the fitted scaler."""
    return scaler.transform(raw_input.reshape(1, -1))


def get_preprocessing_report(df: pd.DataFrame) -> dict:
    """
    Generate a summary report of data quality before preprocessing.
    """
    report = {
        "total_rows":      len(df),
        "total_cols":      len(df.columns),
        "duplicates":      df.duplicated().sum(),
        "missing_total":   df.isnull().sum().sum(),
        "missing_by_col":  df.isnull().sum().to_dict(),
        "missing_pct":     (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes":          df.dtypes.astype(str).to_dict(),
        "class_dist":      df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {},
    }
    return report

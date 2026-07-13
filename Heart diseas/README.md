# ❤️ Heart Disease Prediction System

> **AI-powered cardiovascular risk assessment using advanced machine learning and explainable AI.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

A production-ready, full-stack machine learning application that predicts the probability of heart disease using patient clinical parameters. Features:

- **6 ML models** trained and compared automatically
- **SHAP explainability** for every prediction
- **Interactive visualizations** with Plotly
- **PDF report generation** via ReportLab
- **Dark-themed professional UI** built with Streamlit

---

## 📁 Project Structure

```
Heart diseas/
├── app.py                     ← Main entry point
├── requirements.txt           ← Dependencies
├── README.md                  ← This file
│
├── data/
│   └── heart.csv              ← Auto-downloaded UCI dataset
│
├── models/
│   ├── trainer.py             ← Training, evaluation, model selection
│   └── saved/                 ← Serialized models (auto-created)
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── feature_cols.pkl
│       ├── best_model_name.pkl
│       └── all_results.pkl
│
├── utils/
│   ├── data_loader.py         ← Dataset download + caching
│   ├── preprocessor.py        ← Full preprocessing pipeline
│   ├── metrics.py             ← Evaluation + Plotly charts
│   ├── explainer.py           ← SHAP integration
│   └── report_generator.py   ← PDF generation
│
├── pages/
│   ├── home.py                ← Dashboard
│   ├── data_exploration.py    ← EDA
│   ├── model_training.py      ← Training & evaluation
│   ├── prediction.py          ← Patient prediction
│   └── about.py               ← Project info
│
└── assets/
    └── styles.css             ← Global CSS (dark theme)
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- pip (or conda)

### 2. Clone / navigate to project

```bash
cd "Heart diseas"
```

### 3. Create virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** XGBoost is optional. If installation fails on your platform, remove the `xgboost` line from `requirements.txt` — the app gracefully handles its absence.

### 5. Run the application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 🖥️ Application Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Dashboard with key metrics and navigation |
| 📊 **Data Explorer** | EDA: distributions, correlations, boxplots, heatmaps |
| 🧪 **Model Training** | Train 6 ML models, view leaderboard and ROC curves |
| 🔮 **Prediction** | Input patient data → probability gauge + SHAP + PDF |
| ℹ️ **About** | Dataset info, tech stack, methodology, citations |

---

## 🤖 Machine Learning Models

| Model | Type | SHAP |
|---|---|---|
| Logistic Regression | Linear | LinearExplainer |
| Random Forest | Tree Ensemble | TreeExplainer |
| Decision Tree | Tree | TreeExplainer |
| K-Nearest Neighbors | Instance-based | KernelExplainer |
| Support Vector Machine | Kernel method | KernelExplainer |
| XGBoost *(optional)* | Gradient Boosting | TreeExplainer |

**Model selection**: Best model is automatically chosen by **ROC-AUC** on the held-out test set.

---

## 📊 Dataset

**UCI Heart Disease Dataset (Cleveland)**
- 303 patient records
- 13 clinical features + 1 binary target
- Auto-downloaded from UCI ML Repository on first run

### Features

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | 1 = Male, 0 = Female |
| Chest Pain Type | 0 = Typical Angina, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic |
| Resting BP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| Fasting Blood Sugar | > 120 mg/dl (1=True) |
| Resting ECG | 0 = Normal, 1 = ST-T abnormality, 2 = LVH |
| Max Heart Rate | Maximum heart rate during exercise |
| Exercise Angina | 1 = Yes, 0 = No |
| ST Depression | Oldpeak value |
| ST Slope | 0 = Upsloping, 1 = Flat, 2 = Downsloping |

---

## ☁️ Deployment

### Streamlit Cloud (Free)

1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → set `app.py` as entry point
4. Deploy!

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t heart-disease-predictor .
docker run -p 8501:8501 heart-disease-predictor
```

---

## ⚕️ Medical Disclaimer

> This application is built for **educational and research purposes only**. 
> The predictions generated by this system are **NOT a substitute for professional medical advice, diagnosis, or treatment**.
> Always consult a qualified healthcare provider for medical decisions.

---

## 📜 Citation

```bibtex
@misc{uci_heart_disease,
  author = {Janosi, A. and Steinbrunn, W. and Pfisterer, M. and Detrano, R.},
  title  = {Heart Disease Dataset},
  year   = {1988},
  url    = {https://doi.org/10.24432/C52P4X},
  note   = {UCI Machine Learning Repository}
}
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| Dataset download fails | Check internet connection; dataset is cached after first download |
| XGBoost install error | Remove `xgboost` from `requirements.txt` |
| SHAP slow on KNN/SVM | KernelExplainer is slow by design; switch to Random Forest for speed |
| PDF generation error | Ensure `reportlab` is installed: `pip install reportlab` |
| Port 8501 in use | Run with `streamlit run app.py --server.port 8502` |

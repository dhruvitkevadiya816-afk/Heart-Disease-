@echo off
REM ── Heart Disease Prediction System — GitHub Push Script ──────────────────
REM Run this after Git is installed to push the project to GitHub.

SET REPO_NAME=heart-disease-prediction-system
SET GITHUB_USER=dhruvitkevadiya
SET GIT=C:\Program Files\Git\bin\git.exe
SET GH=C:\Program Files\GitHub CLI\gh.exe

echo [1/5] Initialising git repository...
"%GIT%" init
"%GIT%" config user.name "Dhruv"
"%GIT%" config user.email "%GITHUB_USER%@users.noreply.github.com"

echo [2/5] Adding all files...
"%GIT%" add .
"%GIT%" commit -m "Initial commit: Heart Disease Prediction System"

echo [3/5] Creating GitHub repository...
"%GH%" repo create %REPO_NAME% --public --description "AI-powered Heart Disease Prediction System using Python, Streamlit, Scikit-learn, SHAP" --source=. --remote=origin --push

echo Done! Repo at: https://github.com/%GITHUB_USER%/%REPO_NAME%
pause

# Titanic Survival Prediction — End-to-End ML Pipeline

A production-ready ML application predicting Titanic passenger survival. Covers preprocessing, model training, API, containerization, and deployment.

## Table of Contents
- Overview
- Features
- Tech Stack
- Project Structure
- Installation & Setup
- Usage
- API Reference
- Model Details
- Deployment
- License

## Overview
Uses a **Soft Voting Classifier** (Random Forest + SVM) to predict survival from passenger data (Age, Class, Sex, Fare, etc.). Served via **FastAPI** with an HTML/JS frontend. Emphasizes reproducibility, modularity, scalability, and robustness.

## Features
- Web UI for real-time predictions
- REST API with documentation
- Class + probability output
- One-command automated training
- Frontend & backend input validation

## Tech Stack
- **Python** 3.9+
- **ML**: Scikit-Learn, Pandas, NumPy, Joblib
- **API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, CSS3, Vanilla JS
- **Container**: Docker
- **Tools**: Make, Git

## Project Structure
```text
TITANIC-DATASET/
├── api/index.py
├── config/{env-prod.yaml, env-dev.yaml}
├── data/
├── models/pipeline.pkl
├── src/
│   ├── pipeline/{train_pipeline.py, predict_pipeline.py, preprocessing.py}
│   ├── static/
│   └── templates/
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md

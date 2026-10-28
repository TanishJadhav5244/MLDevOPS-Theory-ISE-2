# MLOps Drift Detection & Continuous Retraining Pipeline with MLflow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/mlflow-3.12.0-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end industry-grade **MLOps Drift Detection and Automated Retraining System** developed for **Tool 1: MLflow Assignment** (FinTech / Banking Credit Risk Default Domain).

Mapped to:
- **CO1 (Level 2 - Understand/Comprehend):** Theoretical concepts across Units 1, 2, and 3 (MLOps Lifecycle, MLflow Tracking, Drift Taxonomy: Covariate Shift, Concept Drift, Prior Probability Shift, PSI, KS-Test).
- **CO2 (Level 3 - Apply/Design/Deploy):** Pipeline design, model tracking, statistical drift monitoring engine, automated Continuous Training (CT), and FastAPI REST deployment.

---

## 📂 Project Architecture

```
d:/Theory ISE 2/
├── run_pipeline.py                 # Master automated execution script (1-Click Run)
├── requirements.txt                # Project dependencies
├── assignment_report_erp_submission.md # Complete Academic Report for ERP upload
├── README.md                       # Documentation & Quickstart
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central paths, feature lists & drift thresholds
│   ├── data_generator.py           # Synthetic FinTech credit risk dataset generator
│   ├── train.py                    # Scikit-learn model training & MLflow tracking
│   ├── drift_engine.py             # PSI, KS-test, Chi-square & plot visualizer
│   ├── monitor_and_retrain.py      # Production monitoring & continuous retraining engine
│   └── app.py                      # FastAPI real-time scoring & drift telemetry service
├── data/                           # Generated CSV datasets
│   ├── baseline_reference.csv      # In-distribution baseline reference data (10k records)
│   ├── production_batch1_no_drift.csv     # Stable production traffic
│   ├── production_batch2_data_drift.csv   # Covariate shift (rate hike & inflation shock)
│   └── production_batch3_concept_drift.csv# Structural concept drift (P(Y|X) shift)
├── artifacts/                      # Model binaries, confusion matrices & drift charts
│   ├── baseline_model.joblib
│   ├── baseline_confusion_matrix.png
│   ├── baseline_roc_curve.png
│   ├── drift_plot_Batch_1_Stable.png
│   ├── drift_plot_Batch_2_DataDrift.png
│   ├── drift_plot_Batch_3_ConceptDrift.png
│   └── drift_report_*.json
└── mlruns/                         # MLflow experiment tracking database & runs
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
Execute the master runner to generate data, train baseline model, monitor drift across 3 production batches, and trigger automated retraining:
```bash
python run_pipeline.py
```

### 3. Launch MLflow Tracking UI
Launch the interactive MLflow UI to inspect logged parameters, metrics, artifact plots, and models:
```bash
python -m mlflow ui --backend-store-uri ./mlruns --port 5000
```
Open your browser at: `http://127.0.0.1:5000`

### 4. Launch FastAPI Deployment Service
Start the real-time inference server:
```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger UI available at: `http://127.0.0.1:8000/docs`

---

## 📊 Summary of Experimental Results

| Production Batch | Data Drift | Concept Drift | Mean PSI | Pred PSI | Label Delta | Accuracy | F1-Score | ROC-AUC | Action Taken |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Batch 1 (Stable)** | **NO** | **NO** | 0.0030 | 0.0220 | 0.0022 | 0.6536 | 0.5792 | 0.6951 | Model continues serving |
| **Batch 2 (Covariate Shift)** | **YES** | **YES** | 1.5086 | 2.9645 | 0.4322 | 0.8448 | 0.9139 | 0.6763 | **Automated CT Retraining Triggered** (New AUC: 0.8005) |
| **Batch 3 (Concept Drift)** | **YES** | **YES** | 0.1566 | 0.7269 | 0.5670 | 0.7616 | 0.8627 | 0.7373 | **Automated CT Retraining Triggered** (New AUC: 0.7910) |

---

## 📑 Full Academic Report

For the complete 20-marks submission text covering theoretical principles of Units 1, 2, 3, mathematical derivations (PSI, KS-test, Chi-square), and ERP upload format, see [assignment_report_erp_submission.md](file:///d:/Theory%20ISE%202/assignment_report_erp_submission.md).

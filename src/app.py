import os
import json
import joblib
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ARTIFACTS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

app = FastAPI(
    title="FinTech Credit Risk MLOps Inference & Drift Service",
    description="Real-time serving API with integrated drift telemetry and model deployment under MLflow.",
    version="1.0.0"
)

# Load latest model pipeline
model_path = ARTIFACTS_DIR / "baseline_model.joblib"
if model_path.exists():
    model = joblib.load(model_path)
else:
    model = None

class LoanApplication(BaseModel):
    applicant_age: int = Field(..., example=35)
    annual_income: float = Field(..., example=65000.0)
    credit_score: float = Field(..., example=710.0)
    debt_to_income_ratio: float = Field(..., example=0.28)
    loan_amount: float = Field(..., example=15000.0)
    interest_rate: float = Field(..., example=10.5)
    employment_length_years: int = Field(..., example=6)
    open_credit_lines: int = Field(..., example=8)
    revolving_utilization_rate: float = Field(..., example=0.25)
    delinquencies_2yrs: int = Field(..., example=0)
    home_ownership: str = Field(..., example="MORTGAGE")
    loan_purpose: str = Field(..., example="debt_consolidation")
    employment_type: str = Field(..., example="salaried")

class PredictionResponse(BaseModel):
    default_prediction: int
    risk_status: str
    default_probability: float
    model_version: str

@app.get("/")
def root():
    return {
        "service": "FinTech Credit Risk MLOps Serving Engine",
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    
    input_data = pd.DataFrame([application.model_dump()])
    prob = float(model.predict_proba(input_data)[0, 1])
    pred = int(prob >= 0.5)
    risk_label = "HIGH RISK (DEFAULT)" if pred == 1 else "LOW RISK (APPROVED)"

    return {
        "default_prediction": pred,
        "risk_status": risk_label,
        "default_probability": round(prob, 4),
        "model_version": "RandomForest_v1.0"
    }

@app.get("/drift_status")
def get_latest_drift_status():
    """
    Returns the latest production drift monitoring status from stored artifacts.
    """
    reports = list(ARTIFACTS_DIR.glob("drift_report_*.json"))
    if not reports:
        return {"message": "No drift monitoring reports found. Run monitor_and_retrain.py first."}
    
    latest_report = sorted(reports)[-1]
    with open(latest_report, "r") as f:
        data = json.load(f)

    return {
        "latest_evaluated_batch": data["drift_summary"]["batch_name"],
        "dataset_drift_detected": data["drift_summary"]["dataset_drift_detected"],
        "mean_psi": data["drift_summary"]["mean_psi"],
        "max_psi": data["drift_summary"]["max_psi"],
        "most_drifted_feature": data["drift_summary"]["most_drifted_feature"],
        "prediction_psi": data["drift_summary"]["prediction_psi"],
        "prediction_drift_detected": data["drift_summary"]["prediction_drift_detected"],
        "label_drift_detected": data["drift_summary"]["label_drift_detected"],
        "performance": data["performance"]
    }

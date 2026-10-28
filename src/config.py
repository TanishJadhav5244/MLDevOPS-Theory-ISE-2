# Configuration and Constants for MLOps Drift Detection Pipeline
import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLFLOW_TRACKING_DIR = BASE_DIR / "mlruns"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

# MLflow Settings
EXPERIMENT_NAME = "Credit_Risk_Drift_Monitoring"
MODEL_NAME = "CreditDefaultRiskModel"
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_TRACKING_DIR.as_posix()}"

# Drift Thresholds (Industry Standards in Banking / Risk Scoring)
# PSI Interpretation:
# PSI < 0.1: No significant shift (Stable)
# 0.1 <= PSI < 0.25: Moderate shift (Monitor closely)
# PSI >= 0.25: Significant shift (Drift detected -> Trigger Retraining)
PSI_MODERATE_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25

# KS-test significance level (alpha)
KS_ALPHA = 0.05

# Feature definitions for FinTech Credit Risk Dataset
NUMERICAL_FEATURES = [
    "applicant_age",
    "annual_income",
    "credit_score",
    "debt_to_income_ratio",
    "loan_amount",
    "interest_rate",
    "employment_length_years",
    "open_credit_lines",
    "revolving_utilization_rate",
    "delinquencies_2yrs"
]

CATEGORICAL_FEATURES = [
    "home_ownership",      # RENT, MORTGAGE, OWN
    "loan_purpose",        # debt_consolidation, credit_card, small_business, home_improvement, education
    "employment_type"      # salaried, self_employed, contract
]

TARGET_COL = "loan_default"  # 0: Good customer, 1: Default

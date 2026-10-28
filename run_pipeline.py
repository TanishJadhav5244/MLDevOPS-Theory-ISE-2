"""
Master Pipeline Runner for MLOps Drift Detection Assignment
Executes:
  1. Synthetic FinTech Dataset Generation (Baseline + 3 Production Drift Batches)
  2. Baseline Model Training & Experiment Tracking in MLflow
  3. Production Drift Monitoring (PSI, KS-Test, Prediction Drift, Label Drift)
  4. Automated Continuous Retraining & Logging to MLflow
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_generator import generate_and_save_all_datasets
from src.train import train_baseline_model
from src.monitor_and_retrain import run_production_drift_monitoring

def main():
    print("=" * 70)
    print("   MLOps ASSIGNMENT - TOOL 1: MLflow DRIFT DETECTION & RETRAINING   ")
    print("   Domain: FinTech Credit Risk & Loan Default Prediction             ")
    print("=" * 70)
    
    start_time = time.time()

    # Step 1: Generate Datasets
    print("\n>>> STEP 1: Generating Datasets (Reference & Production Drift Batches)...")
    generate_and_save_all_datasets()

    # Step 2: Train Baseline Model
    print("\n>>> STEP 2: Training Baseline Model & Tracking in MLflow...")
    train_baseline_model()

    # Step 3: Drift Monitoring & Automated Continuous Training
    print("\n>>> STEP 3: Running Drift Monitoring & Continuous Retraining Engine...")
    run_production_drift_monitoring()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f" [+] Pipeline executed successfully in {elapsed} seconds!")
    print(" [+] Artifacts saved to: ./artifacts")
    print(" [+] MLflow runs saved to: ./mlruns")
    print("=" * 70)
    print("\nTo explore the MLflow UI, run:")
    print("    python -m mlflow ui --backend-store-uri ./mlruns --port 5000")
    print("\nTo launch the FastAPI Inference & Drift API, run:")
    print("    python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload")
    print("=" * 70)

if __name__ == "__main__":
    main()

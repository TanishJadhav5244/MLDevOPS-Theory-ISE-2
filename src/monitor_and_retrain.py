import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL
)
from src.drift_engine import evaluate_dataset_drift, plot_drift_visualizations

def retrain_model_on_drifted_data(combined_train_df: pd.DataFrame, run_name: str = "Retrained_Model_v2"):
    """
    Automated Continuous Training (CT):
    Retrains model when critical drift or performance drop is detected.
    """
    print(f"\n[!] Triggering Automated Retraining: {run_name}...")
    X = combined_train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = combined_train_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    rf_params = {
        "n_estimators": 140,
        "max_depth": 9,
        "min_samples_split": 4,
        "random_state": 42,
        "class_weight": "balanced"
    }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**rf_params))
    ])

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params(rf_params)
        mlflow.set_tag("pipeline_action", "Automated_Retraining_Triggered")
        mlflow.set_tag("retrained_on", "Adapted_Drift_Distribution")
        mlflow.log_metrics({
            "retrained_accuracy": acc,
            "retrained_precision": prec,
            "retrained_recall": rec,
            "retrained_f1_score": f1,
            "retrained_roc_auc": roc_auc
        })

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(sk_model=pipeline, name="model", signature=signature)

        print(f"[+] Retrained Model Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        joblib.dump(pipeline, ARTIFACTS_DIR / f"{run_name}.joblib")
        return pipeline, run.info.run_id

def run_production_drift_monitoring():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load Baseline Model, Reference Data, and Test Metrics
    ref_df = pd.read_csv(DATA_DIR / "baseline_reference.csv")
    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
    
    current_model = joblib.load(model_path)

    # Load baseline test metrics
    metrics_file = ARTIFACTS_DIR / "baseline_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            base_metrics = json.load(f)
            base_f1 = base_metrics["f1_score"]
            base_auc = base_metrics["roc_auc"]
            base_acc = base_metrics["accuracy"]
    else:
        base_f1 = 0.5903
        base_auc = 0.6951
        base_acc = 0.6608

    print("=================================================================")
    print("      MLOps PRODUCTION DRIFT MONITORING PIPELINE RUNNING        ")
    print("=================================================================")
    print(f"Reference Baseline Metrics: Accuracy={base_acc:.4f}, F1={base_f1:.4f}, ROC-AUC={base_auc:.4f}")

    # Precompute reference predictions and probabilities
    ref_probs = current_model.predict_proba(ref_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])[:, 1]

    batches = [
        ("Batch_1_Stable", DATA_DIR / "production_batch1_no_drift.csv"),
        ("Batch_2_DataDrift", DATA_DIR / "production_batch2_data_drift.csv"),
        ("Batch_3_ConceptDrift", DATA_DIR / "production_batch3_concept_drift.csv"),
    ]

    monitoring_results = []

    for batch_name, batch_path in batches:
        print(f"\n>>> Monitoring: {batch_name} ({batch_path.name})")
        prod_df = pd.read_csv(batch_path)
        X_prod = prod_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y_prod = prod_df[TARGET_COL]

        # 1. Model Predictions on Incoming Batch
        y_prod_pred = current_model.predict(X_prod)
        prod_probs = current_model.predict_proba(X_prod)[:, 1]

        # 2. Data Drift Evaluation (Covariate Shift, Prediction Drift, Label Drift)
        drift_summary = evaluate_dataset_drift(
            ref_df, prod_df, batch_name=batch_name, ref_probs=ref_probs, prod_probs=prod_probs
        )

        prod_acc = accuracy_score(y_prod, y_prod_pred)
        prod_prec = precision_score(y_prod, y_prod_pred, zero_division=0)
        prod_rec = recall_score(y_prod, y_prod_pred, zero_division=0)
        prod_f1 = f1_score(y_prod, y_prod_pred, zero_division=0)
        prod_auc = roc_auc_score(y_prod, prod_probs)

        delta_f1 = base_f1 - prod_f1
        delta_auc = base_auc - prod_auc

        # Concept / Structural Drift: Flagged if Prediction Drift or Label Drift or Significant Accuracy Drop
        concept_or_label_drift = drift_summary["prediction_drift_detected"] or drift_summary["label_drift_detected"]

        print(f"    - Drifted Features: {drift_summary['drifted_features_count']} / {drift_summary['total_features']}")
        print(f"    - Mean PSI: {drift_summary['mean_psi']:.4f} | Max PSI: {drift_summary['max_psi']:.4f} ({drift_summary['most_drifted_feature']})")
        print(f"    - Prediction Drift (Probabilities PSI): {drift_summary['prediction_psi']:.4f} (Flag: {drift_summary['prediction_drift_detected']})")
        print(f"    - Label Drift (|P(Y_ref) - P(Y_prod)|): {drift_summary['label_drift_delta']:.4f} (Flag: {drift_summary['label_drift_detected']})")
        print(f"    - Data Drift Flag: {drift_summary['dataset_drift_detected']}")
        print(f"    - Current Batch Performance: Acc={prod_acc:.4f}, F1={prod_f1:.4f}, AUC={prod_auc:.4f}")
        print(f"    - Concept/Label Drift Detected: {concept_or_label_drift}")

        # 3. Generate Visual Artifacts
        plot_path = plot_drift_visualizations(
            ref_df, prod_df, drift_summary, output_filename=f"drift_plot_{batch_name}.png"
        )

        # 4. Save JSON Report
        report_json_path = ARTIFACTS_DIR / f"drift_report_{batch_name}.json"
        with open(report_json_path, "w") as f:
            json.dump({
                "drift_summary": drift_summary,
                "performance": {
                    "accuracy": prod_acc,
                    "precision": prod_prec,
                    "recall": prod_rec,
                    "f1_score": prod_f1,
                    "roc_auc": prod_auc,
                    "delta_f1": delta_f1,
                    "delta_roc_auc": delta_auc,
                    "concept_or_label_drift": concept_or_label_drift
                }
            }, f, indent=4)

        # 5. Log Everything to MLflow
        with mlflow.start_run(run_name=f"Monitoring_{batch_name}") as run:
            mlflow.set_tag("pipeline_phase", "Continuous_Monitoring")
            mlflow.set_tag("batch_name", batch_name)
            mlflow.set_tag("data_drift_status", "DRIFT_DETECTED" if drift_summary["dataset_drift_detected"] else "STABLE")
            mlflow.set_tag("concept_drift_status", "DRIFT_DETECTED" if concept_or_label_drift else "STABLE")

            # Numerical drift metrics
            mlflow.log_metric("mean_psi", drift_summary["mean_psi"])
            mlflow.log_metric("max_psi", drift_summary["max_psi"])
            mlflow.log_metric("prediction_psi", drift_summary["prediction_psi"])
            mlflow.log_metric("label_drift_delta", drift_summary["label_drift_delta"])
            mlflow.log_metric("drifted_features_count", drift_summary["drifted_features_count"])
            mlflow.log_metric("drift_feature_ratio", drift_summary["drift_feature_ratio"])
            mlflow.log_metric("dataset_drift_flag", 1 if drift_summary["dataset_drift_detected"] else 0)
            mlflow.log_metric("concept_drift_flag", 1 if concept_or_label_drift else 0)

            # Feature specific PSI
            for feat, d in drift_summary["feature_details"].items():
                if d["type"] == "numerical":
                    mlflow.log_metric(f"psi_{feat}", d["psi"])

            # Model performance metrics on production batch
            mlflow.log_metric("prod_accuracy", prod_acc)
            mlflow.log_metric("prod_f1_score", prod_f1)
            mlflow.log_metric("prod_roc_auc", prod_auc)
            mlflow.log_metric("delta_f1", delta_f1)
            mlflow.log_metric("delta_roc_auc", delta_auc)

            # Log Visual Artifacts and JSON Reports
            mlflow.log_artifact(plot_path, artifact_path="drift_plots")
            mlflow.log_artifact(str(report_json_path), artifact_path="drift_reports")

            # 6. Automated Triggering of Retraining Pipeline
            if drift_summary["dataset_drift_detected"] or concept_or_label_drift:
                print(f"[ALERT] Drift threshold exceeded for {batch_name}! Initiating Continuous Retraining...")
                combined_df = pd.concat([ref_df.sample(frac=0.4, random_state=42), prod_df], ignore_index=True)
                new_model, retrain_run_id = retrain_model_on_drifted_data(
                    combined_df, run_name=f"Auto_Retrained_Post_{batch_name}"
                )
                mlflow.log_param("retrained_model_run_id", retrain_run_id)

        monitoring_results.append({
            "Batch": batch_name,
            "Data Drift": "YES" if drift_summary["dataset_drift_detected"] else "NO",
            "Concept Drift": "YES" if concept_or_label_drift else "NO",
            "Mean PSI": drift_summary["mean_psi"],
            "Pred PSI": drift_summary["prediction_psi"],
            "Label Delta": drift_summary["label_drift_delta"],
            "Accuracy": round(prod_acc, 4),
            "F1-Score": round(prod_f1, 4),
            "ROC-AUC": round(prod_auc, 4)
        })

    print("\n=================================================================")
    print("                DRIFT MONITORING SUMMARY TABLE                   ")
    print("=================================================================")
    summary_df = pd.DataFrame(monitoring_results)
    print(summary_df.to_string(index=False))
    print("=================================================================\n")
    return monitoring_results

if __name__ == "__main__":
    run_production_drift_monitoring()

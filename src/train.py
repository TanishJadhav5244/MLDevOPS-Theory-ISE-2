import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_NAME,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL
)

def train_baseline_model():
    print("[*] Setting up MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load Data
    data_path = DATA_DIR / "baseline_reference.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Baseline data not found at {data_path}. Run data_generator.py first.")

    df = pd.read_csv(data_path)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Build Preprocessor & Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    rf_params = {
        "n_estimators": 120,
        "max_depth": 8,
        "min_samples_split": 5,
        "random_state": 42,
        "class_weight": "balanced"
    }

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**rf_params))
    ])

    print("[*] Training Baseline Model within MLflow run...")
    with mlflow.start_run(run_name="Baseline_Model_v1") as run:
        run_id = run.info.run_id
        print(f"[+] Active MLflow Run ID: {run_id}")

        # Train
        model_pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("\n--- Baseline Model Evaluation Metrics ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("-----------------------------------------\n")

        # Log Parameters & Tags
        mlflow.log_params(rf_params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("stage", "Production_Baseline")
        mlflow.set_tag("domain", "FinTech_Credit_Risk")

        # Log Metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        # Generate & Log Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
        plt.title("Baseline Model - Confusion Matrix")
        cm_path = ARTIFACTS_DIR / "baseline_confusion_matrix.png"
        plt.savefig(cm_path, dpi=180, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Generate & Log ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"ROC curve (area = {roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Baseline ROC-AUC Curve")
        ax_roc.legend(loc="lower right")
        roc_path = ARTIFACTS_DIR / "baseline_roc_curve.png"
        plt.savefig(roc_path, dpi=180, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # Infer signature and log model
        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model_pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            signature=signature,
            input_example=input_example
        )

        # Save model and baseline reference metrics
        saved_model_path = ARTIFACTS_DIR / "baseline_model.joblib"
        joblib.dump(model_pipeline, saved_model_path)

        import json
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        with open(ARTIFACTS_DIR / "baseline_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact(str(ARTIFACTS_DIR / "baseline_metrics.json"), artifact_path="metrics")

        print(f"[+] Model pipeline serialized locally to {saved_model_path}")
        print(f"[+] MLflow experiment run completed. Run ID: {run_id}")

        return model_pipeline, run_id

if __name__ == "__main__":
    train_baseline_model()

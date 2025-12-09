# logistic_sklearn_ohe.py
import time
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Constants
TARGET = "Outcome"
CSV_PATH = r"../examples/Logistic_regression_5.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUT_DIR = "models"
MODEL_FILENAME = "logistic_pipeline.joblib"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load dataset
    data = pd.read_csv(CSV_PATH)

    if TARGET not in data.columns:
        raise ValueError(f"TARGET '{TARGET}' not found. Available columns: {list(data.columns)}")

    # Separate features & target
    X = data.drop(columns=[TARGET])
    y = data[TARGET].values.ravel()

    # Detect categorical columns (include "Gender")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical Columns:", categorical_cols)
    print("Numeric Columns:", numeric_cols)

    # Preprocessor: OHE for categorical, scaling for numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Final pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train time
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Prediction time
    t1 = time.perf_counter()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t1

    per_sample = predict_time / len(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
print(type(arr))
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTiming:")
    print(f"Training time       : {train_time:.6f} sec")
    print(f"Prediction time     : {predict_time:.6f} sec")
    print(f"Per-sample predict  : {per_sample:.9f} sec/sample")

    # Save model
    model_path = Path(OUT_DIR) / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model → {model_path.resolve()}")


if __name__ == "__main__":
    main()


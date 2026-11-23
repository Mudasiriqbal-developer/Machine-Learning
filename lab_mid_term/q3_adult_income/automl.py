# AutoML pipeline for income classification using TPOT
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import tpot
from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

MAX_MINUTES = 2


def main():
    # 1. load dataset splits
    data_dir = "data" if os.path.exists("data/X_train.csv") else "q3_adult_income/data"

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # 2. build ColumnTransformer (dense output for TPOT)
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # 3. fit preprocessor on train only and transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # 4. configure TPOT
    tpot_version = getattr(tpot, "__version__", "0.12")
    print(f"Using TPOT version: {tpot_version}")

    tpot_model = TPOTClassifier(
        max_time_mins=MAX_MINUTES,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=4,
        verbosity=2
    )

    # 5. run optimization search
    print(f"Starting TPOT search (time limit: {MAX_MINUTES} minutes)...")
    start_time = time.time()
    tpot_model.fit(X_train_prep, y_train)
    train_time_sec = round(time.time() - start_time, 2)
    print(f"TPOT search finished in {train_time_sec}s")

    # 6. assemble final pipeline with preprocessor and best model
    best_tpot_pipeline = tpot_model.fitted_pipeline_

    final_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", best_tpot_pipeline)
    ])

    # 7. evaluate on raw X_test
    y_pred = final_pipeline.predict(X_test)

    if hasattr(final_pipeline, "predict_proba"):
        try:
            y_prob = final_pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = final_pipeline.decision_function(X_test)
    elif hasattr(final_pipeline, "decision_function"):
        y_prob = final_pipeline.decision_function(X_test)
    else:
        y_prob = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    # 8. save final pipeline and metrics
    models_dir = "models" if os.path.exists("models") else "q3_adult_income/models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "automl.pkl")
    joblib.dump(final_pipeline, model_path)
    print(f"Saved final pipeline to {model_path}")

    best_pipeline_str = str(best_tpot_pipeline)

    results = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "train_time_sec": train_time_sec,
        "best_pipeline": best_pipeline_str
    }

    results_path = os.path.join(models_dir, "automl_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {results_path}")

    # 9. print summary
    print("\n--- AutoML Evaluation Results ---")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")
    print(f"Train Time:     {train_time_sec}s")
    print(f"\nBest Pipeline:\n{best_pipeline_str}")


if __name__ == "__main__":
    main()

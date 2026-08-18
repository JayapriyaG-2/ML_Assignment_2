import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DATA_PATH = os.path.join(ROOT, "data", "pd_speech_features.csv")


def load_data():
    # Original file has a 2-row header (category group row + actual names)
    df = pd.read_csv(RAW_DATA_PATH, header=1)

    groups = df["id"]
    y = df["class"].astype(int)
    X = df.drop(columns=["id", "class"])
    feature_names = list(X.columns)
    return X, y, groups, feature_names


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred
    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_score), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }


def main():
    X, y, groups, feature_names = load_data()

    # Patient-level split: all recordings from one patient stay together
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(model, X_test_scaled, y_test)
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(HERE, f"{fname}.joblib"))

    joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))
    with open(os.path.join(HERE, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "ML Model Name"
    metrics_df = metrics_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics_df.to_csv(os.path.join(HERE, "metrics_comparison.csv"))

    n_train_patients = groups.iloc[train_idx].nunique()
    n_test_patients = groups.iloc[test_idx].nunique()

    print("\n=== Evaluation Metrics on Held-out (Patient-level) Test Set ===\n")
    print(metrics_df.to_string())
    print(f"\nTrain: {len(X_train)} recordings from {n_train_patients} patients")
    print(f"Test:  {len(X_test)} recordings from {n_test_patients} patients")
    print(f"Saved test_data.csv with {len(test_df)} rows to project root.")
    print("Saved trained models + scaler to model/ folder.")


if __name__ == "__main__":
    main()

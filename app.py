import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Parkinson's Speech Classifier", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib",
}


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.joblib")


@st.cache_resource
def load_feature_names():
    with open("model/feature_names.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("Parkinson's Disease Classification — Model Comparison App")
    st.caption(
        "Dataset: UCI Parkinson's Disease Classification (speech features) — "
        "754 acoustic/voice-signal features extracted from sustained vowel "
        "recordings. Binary classification (0 = Healthy, 1 = Parkinson's)."
    )

    feature_names = load_feature_names()
    scaler = load_scaler()

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Controls")

    uploaded_file = st.sidebar.file_uploader(
        "Upload test data (CSV)", type=["csv"],
        help=f"Must contain the {len(feature_names)} feature columns plus a "
             "'target' column (0/1). Use the provided test_data.csv."
    )

    model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Required columns:** {len(feature_names)} acoustic features "
        "(baseline, MFCC, wavelet, TQWT groups) + `target`.\n\n"
        "See `test_data.csv` for the exact column set — too many to list here."
    )

    if uploaded_file is None:
        st.info("👈 Upload a test CSV from the sidebar to get started "
                 "(you can use the provided `test_data.csv`).")
        st.subheader("Expected column count")
        st.write(f"{len(feature_names)} feature columns + `target` "
                 f"({len(feature_names) + 1} columns total)")
        st.code(", ".join(feature_names[:15]) + ", ... (+{} more)".format(
            len(feature_names) - 15))
        return

    # ---------------- Load & validate data ----------------
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        return

    missing_cols = [c for c in feature_names + ["target"] if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing {len(missing_cols)} required column(s), "
                 f"e.g.: {missing_cols[:5]}")
        return

    X = df[feature_names]
    y_true = df["target"].astype(int)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names, index=X.index)

    # ---------------- Load model & predict ----------------
    model = load_model(MODEL_FILES[model_name])
    y_pred = model.predict(X_scaled)
    y_score = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    metrics = compute_metrics(y_true, y_pred, y_score)

    # ---------------- Display ----------------
    st.subheader(f"📊 Results — {model_name}")

    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    st.markdown("### Predictions preview")
    preview = df[["target"]].copy() if "target" in df.columns else pd.DataFrame()
    preview["Predicted"] = y_pred
    preview["Predicted Probability (PD=1)"] = np.round(y_score, 4)
    st.dataframe(preview.head(20), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Healthy (0)", "Parkinson's (1)"],
                    yticklabels=["Healthy (0)", "Parkinson's (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("---")
    st.markdown("### All-Model Comparison")
    comp_rows = {}
    for name, path in MODEL_FILES.items():
        m = load_model(path)
        pred = m.predict(X_scaled)
        score = m.predict_proba(X_scaled)[:, 1] if hasattr(m, "predict_proba") else pred
        comp_rows[name] = compute_metrics(y_true, pred, score)
    comp_df = pd.DataFrame(comp_rows).T.round(4)
    comp_df.index.name = "Model"
    st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()

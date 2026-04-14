"""
model.py — Data Scientist reusable functions
Traffic Accident Severity Classifier | Project 2
Day 3: Baseline Decision Tree
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# ── Constants ────────────────────────────────────────────────────────────────

PROCESSED_DATA_PATH  = "data/processed/cleaned_data.csv"
CLASS_WEIGHTS_PATH   = "data/processed/class_weights.json"
OUTPUTS_DIR          = "outputs"
FIGURES_DIR          = os.path.join(OUTPUTS_DIR, "figures")
TARGET_COL           = "Severity"
RANDOM_STATE         = 42
TEST_SIZE            = 0.2


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_processed(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned, feature-engineered dataset produced by the DE pipeline."""
    print(f"Loading processed data from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df[TARGET_COL].value_counts().sort_index()}\n")
    return df


def load_class_weights(path: str = CLASS_WEIGHTS_PATH) -> dict:
    """Load the class weights computed by the DE (handle_class_imbalance)."""
    with open(path) as f:
        weights = json.load(f)
    # sklearn expects int keys for class_weight dict
    return {int(k): v for k, v in weights.items()}


# ── Train / Test Split ────────────────────────────────────────────────────────

def split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    Stratified 80/20 split preserving severity class ratios.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ── Decision Tree ─────────────────────────────────────────────────────────────

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: dict | None = None,
    max_depth: int = 10,
) -> DecisionTreeClassifier:
    """
    Train a baseline Decision Tree with class weights.

    Parameters
    ----------
    max_depth : int
        Limits tree depth to prevent memory explosion on 2.8M rows.
        Start at 10; tune in Day 4 if needed.
    class_weight : dict | None
        Pass the weights from load_class_weights() or use 'balanced'.
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        min_samples_leaf=50,   # avoids overly specific leaves on large dataset
    )
    print(f"Training Decision Tree (max_depth={max_depth})...")
    clf.fit(X_train, y_train)
    print("  Done.\n")
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    clf,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Decision Tree",
) -> dict:
    """
    Print classification report and return a dict of key metrics.
    Covers: per-class precision, recall, F1 + macro/weighted averages.
    """
    y_pred = clf.predict(X_test)

    print(f"\n{'='*60}")
    print(f"  Evaluation — {model_name}")
    print(f"{'='*60}")
    report = classification_report(
        y_test, y_pred,
        target_names=[f"Severity {i}" for i in sorted(y_test.unique())],
    )
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}\n")

    return {
        "model": model_name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred,
    }


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Decision Tree",
    save: bool = True,
) -> None:
    """Plot and optionally save a normalized confusion matrix heatmap."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    labels = [f"Severity {i}" for i in sorted(y_test.unique())]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (normalized)")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
        fig.savefig(path, dpi=150)
        print(f"  Confusion matrix saved to: {path}")

    plt.show()


# ── Feature Importance Plot ───────────────────────────────────────────────────

def plot_feature_importance(
    clf,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "Decision Tree",
    save: bool = True,
) -> None:
    """Bar chart of the top_n most important features."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features[::-1], top_importances[::-1], color="steelblue")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"feature_importance_{model_name.replace(' ', '_').lower()}.png")
        fig.savefig(path, dpi=150)
        print(f"  Feature importance plot saved to: {path}")

    plt.show()


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_model(clf, filename: str = "decision_tree.pkl") -> None:
    """Pickle the trained model to outputs/ for Day 4 comparison."""
    import pickle
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {path}")


def load_model(filename: str = "decision_tree.pkl"):
    """Load a pickled model from outputs/."""
    import pickle
    path = os.path.join(OUTPUTS_DIR, filename)
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"  Model loaded from: {path}")
    return clf
"""
train.py — Day 3: Baseline Decision Tree
Traffic Accident Severity Classifier | Project 2

Run from project root:
    python train.py

Outputs (all in outputs/):
    predictions.csv       — test set rows with y_true and y_pred columns
    metrics_summary.csv   — per-class precision, recall, F1 + macro/weighted
    class_weights.json    — loaded from data/processed/ (written by DE pipeline)
    decision_tree.pkl     — saved model for Day 4 RF comparison
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"
CLASS_WEIGHTS_PATH  = "data/processed/class_weights.json"
OUTPUT_DIR          = "outputs"
RANDOM_STATE        = 42
TEST_SIZE           = 0.20
MAX_DEPTH           = 10       # safe cap for 2.8M rows; tune on Day 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    print(f"Loading: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Severity distribution:\n{df['Severity'].value_counts().sort_index()}\n")
    return df


def load_class_weights():
    with open(CLASS_WEIGHTS_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def split(df):
    X = df.drop(columns=["Severity"])
    y = df["Severity"]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def train(X_train, y_train, class_weights):
    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        class_weight=class_weights,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
    )
    print(f"Training Decision Tree (max_depth={MAX_DEPTH}, min_samples_leaf=50)...")
    clf.fit(X_train, y_train)
    print("  Done.\n")
    return clf


def evaluate_and_save(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    # ── Classification report → CSV ───────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=[f"Severity {i}" for i in sorted(y_test.unique())],
        output_dict=True,
    )
    metrics_df = pd.DataFrame(report).transpose().reset_index()
    metrics_df.rename(columns={"index": "class"}, inplace=True)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved  → {metrics_path}")

    # ── Predictions → CSV (for visualize.py) ─────────────────────────────────
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred

    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved → {pred_path}")

    # ── Feature importances → CSV ─────────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fi_path = os.path.join(OUTPUT_DIR, "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importances → {fi_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    macro_f1    = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\n{'='*50}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"{'='*50}\n")
    

    return y_pred


def save_model(clf):
    path = os.path.join(OUTPUT_DIR, "decision_tree.pkl")
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df             = load_data()
    class_weights  = load_class_weights()
    X_train, X_test, y_train, y_test = split(df)
    clf            = train(X_train, y_train, class_weights)
    evaluate_and_save(clf, X_test, y_test)
    save_model(clf)

    print("\nDone. Run  python visualize.py  to generate all charts.")


if __name__ == "__main__":
    main()
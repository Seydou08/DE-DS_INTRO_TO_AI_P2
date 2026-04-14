"""
visualize.py — Generate all charts from train.py outputs
Traffic Accident Severity Classifier | Project 2

Run AFTER train.py:
    python visualize.py

Reads from outputs/:
    predictions.csv
    metrics_summary.csv
    feature_importances.csv

Saves PNGs to outputs/figures/:
    01_class_distribution.png
    02_metrics_summary.png
    03_confusion_matrix.png
    04_feature_importances.png
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Consistent palette across all charts
PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Load ──────────────────────────────────────────────────────────────────────

pred_df    = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))
metrics_df = pd.read_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"))
fi_df      = pd.read_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"))

y_true = pred_df["y_true"]
y_pred = pred_df["y_pred"]
severity_labels = [f"Severity {i}" for i in sorted(y_true.unique())]


# ── Chart 1: Class Distribution ───────────────────────────────────────────────

def plot_class_distribution():
    dist = y_true.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"Severity {i}" for i in dist.index],
        dist.values,
        color=PALETTE,
        edgecolor="white",
        width=0.6,
    )
    for bar, val in zip(bars, dist.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dist.max() * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Test Set — Severity Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Severity Level", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "01_class_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Chart 2: Metrics Summary (Precision / Recall / F1 per class) ──────────────

def plot_metrics_summary():
    # Filter to per-class rows only (exclude macro avg, weighted avg, accuracy)
    per_class = metrics_df[metrics_df["class"].str.startswith("Severity")].copy()

    x = np.arange(len(per_class))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, per_class["precision"], width, label="Precision", color="#4878CF", edgecolor="white")
    ax.bar(x,         per_class["recall"],    width, label="Recall",    color="#6ACC65", edgecolor="white")
    ax.bar(x + width, per_class["f1-score"],  width, label="F1-Score",  color="#D65F5F", edgecolor="white")

    # Annotate macro F1 from summary rows
    macro_row = metrics_df[metrics_df["class"] == "macro avg"]
    if not macro_row.empty:
        macro_f1 = macro_row["f1-score"].values[0]
        ax.axhline(macro_f1, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Macro F1 = {macro_f1:.3f}")

    ax.set_title("Decision Tree — Per-Class Metrics", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Severity Level", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(per_class["class"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "02_metrics_summary.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Chart 3: Confusion Matrix ─────────────────────────────────────────────────

def plot_confusion_matrix():
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=severity_labels,
        yticklabels=severity_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_title("Decision Tree — Confusion Matrix (normalized)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "03_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Chart 4: Feature Importances ──────────────────────────────────────────────

def plot_feature_importances(top_n: int = 20):
    top = fi_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top["feature"], top["importance"], color="#4878CF", edgecolor="white")

    # Label each bar
    for bar, val in zip(bars, top["importance"]):
        ax.text(
            val + top["importance"].max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8,
        )

    ax.set_title(f"Decision Tree — Top {top_n} Feature Importances (Gini)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_ylabel("")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "04_feature_importances.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating charts...\n")
    plot_class_distribution()
    plot_metrics_summary()
    plot_confusion_matrix()
    plot_feature_importances(top_n=20)
    print(f"\nAll charts saved to: {FIGURES_DIR}/")
"""
Visualizations for the diabetes risk analysis project.

Run: python src/visualize.py
Writes PNG images to results/figures/. Nothing here changes the data or the
models; this script only draws pictures of what data_prep.py, eda.py, and
model.py already computed, so you can SEE the effect of each stage rather
than just reading numbers.

Produces:
  1. missing_values_before_cleaning.png - how many zero placeholders each
     affected measurement had before cleaning.
  2. before_after_cleaning.png - compact before/after histograms for the
     columns most affected by disguised missing values.
  3. correlation_heatmap.png - how strongly each measurement relates to
     every other measurement and to the diabetes outcome.
  4. roc_curve.png - the "before and after prediction" picture: how well
     each model separates diabetic from non-diabetic patients, compared to
     random guessing.
  5. confusion_matrices.png - for each model, how many predictions were
     right vs. wrong, broken down by type of error.
  6. threshold_tradeoff.png - precision/recall at several decision cutoffs.
  7. feature_importance.png - which measurements each model leaned on most.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
)

from data_prep import ZERO_AS_MISSING, load_raw, load_clean
from model import FEATURE_COLS, build_logistic_pipeline, build_rf_pipeline, split_data

FIG_DIR = Path("results/figures")
CLEANING_DETAIL_COLS = ["skin_thickness", "insulin", "bmi", "blood_pressure"]

DISPLAY_NAMES = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "Blood pressure",
    "skin_thickness": "Skin thickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetes_pedigree": "Family history score",
    "age": "Age",
    "diabetes": "Diabetes",
    "insulin_missing": "Insulin missing",
    "skin_thickness_missing": "Skin thickness missing",
}


def pretty_name(column: str) -> str:
    return DISPLAY_NAMES.get(column, column.replace("_", " ").title())


def plot_before_after_cleaning(raw: pd.DataFrame, clean: pd.DataFrame):
    fig, axes = plt.subplots(len(CLEANING_DETAIL_COLS), 2, figsize=(10, 10))
    for i, col in enumerate(CLEANING_DETAIL_COLS):
        axes[i, 0].hist(raw[col], bins=30, color="#c0392b")
        axes[i, 0].set_title(f"{pretty_name(col)} before cleaning")
        axes[i, 0].set_ylabel("Patients")

        axes[i, 1].hist(clean[col], bins=30, color="#2980b9")
        axes[i, 1].set_title(f"{pretty_name(col)} after cleaning")

    fig.suptitle("Effect of replacing impossible zero values", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "before_after_cleaning.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_missing_values_before_cleaning(raw: pd.DataFrame):
    missing_counts = (raw[ZERO_AS_MISSING] == 0).sum().sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [pretty_name(col) for col in missing_counts.index]
    bars = ax.barh(labels, missing_counts.values, color="#6c5ce7")
    ax.set_title("Zero placeholders before cleaning")
    ax.set_xlabel("Rows with impossible zero values")
    ax.set_xlim(0, max(missing_counts.values) * 1.15)
    for bar, count in zip(bars, missing_counts.values):
        pct = count / len(raw) * 100
        ax.text(count + 4, bar.get_y() + bar.get_height() / 2, f"{count} ({pct:.0f}%)",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missing_values_before_cleaning.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(clean: pd.DataFrame):
    numeric_cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                     "insulin", "bmi", "diabetes_pedigree", "age", "diabetes"]
    corr = clean[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels([pretty_name(col) for col in numeric_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels([pretty_name(col) for col in numeric_cols])
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="correlation (-1 to 1)")
    ax.set_title("Correlation between measurements")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_and_confusion(X_train, X_test, y_train, y_test):
    log_reg = build_logistic_pipeline().fit(X_train, y_train)
    rf = build_rf_pipeline().fit(X_train, y_train)

    # ROC curve: the "after prediction" picture -- how well each model
    # separates diabetic from non-diabetic patients across every possible
    # decision threshold, not just the default 0.5 cutoff
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(log_reg, X_test, y_test, ax=ax, name="Logistic Regression")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices: exactly which patients each model got right/wrong
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, model) in zip(axes, [("Logistic Regression", log_reg), ("Random Forest", rf)]):
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.suptitle("Predicted vs. actual outcomes on the test set", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return log_reg, rf


def plot_threshold_tradeoff(models, X_test, y_test):
    thresholds = np.array([0.30, 0.40, 0.50])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, (name, model) in zip(axes, models):
        proba = model.predict_proba(X_test)[:, 1]
        precision = []
        recall = []
        for threshold in thresholds:
            pred = (proba >= threshold).astype(int)
            precision.append(precision_score(y_test, pred, zero_division=0))
            recall.append(recall_score(y_test, pred, zero_division=0))

        x = np.arange(len(thresholds))
        width = 0.36
        ax.bar(x - width / 2, recall, width, label="Recall", color="#d35400")
        ax.bar(x + width / 2, precision, width, label="Precision", color="#2980b9")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
        ax.set_xlabel("Decision threshold")
        ax.set_ylim(0, 1)
        for xpos, values in [(x - width / 2, recall), (x + width / 2, precision)]:
            for px, val in zip(xpos, values):
                ax.text(px, val + 0.02, f"{val:.0%}", ha="center", fontsize=8)

    axes[0].set_ylabel("Score")
    axes[1].legend(loc="lower right")
    fig.suptitle("Precision/recall tradeoff at different thresholds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_tradeoff.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(log_reg, rf):
    importances = rf.named_steps["clf"].feature_importances_
    coefs = log_reg.named_steps["clf"].coef_[0]

    order = np.argsort(importances)[::-1]
    features_sorted = [pretty_name(FEATURE_COLS[i]) for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(features_sorted[::-1], importances[order][::-1], color="#27ae60")
    axes[0].set_title("Random Forest feature importance")
    axes[0].set_xlabel("Importance")

    coef_order = np.argsort(np.abs(coefs))[::-1]
    coef_features_sorted = [pretty_name(FEATURE_COLS[i]) for i in coef_order]
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in coefs[coef_order]]
    axes[1].barh(coef_features_sorted[::-1], coefs[coef_order][::-1], color=colors[::-1])
    axes[1].set_title("Logistic Regression coefficients")
    axes[1].set_xlabel("Coefficient")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    clean = load_clean()

    print("Plotting missing zero placeholders...")
    plot_missing_values_before_cleaning(raw)

    print("Plotting before/after cleaning...")
    plot_before_after_cleaning(raw, clean)

    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(clean)

    print("Training models for ROC / confusion matrix plots...")
    X_train, X_test, y_train, y_test = split_data(clean)
    log_reg, rf = plot_roc_and_confusion(X_train, X_test, y_train, y_test)

    print("Plotting precision/recall threshold tradeoff...")
    plot_threshold_tradeoff(
        [("Logistic Regression", log_reg), ("Random Forest", rf)],
        X_test,
        y_test,
    )

    print("Plotting feature importance...")
    plot_feature_importance(log_reg, rf)

    print(f"\nDone. 7 images saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

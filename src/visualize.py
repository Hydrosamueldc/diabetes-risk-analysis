"""
Visualizations for the diabetes risk analysis project.

Run: python src/visualize.py
Writes PNG images to results/figures/. Nothing here changes the data or the
models; this script only draws pictures of what data_prep.py, eda.py, and
model.py already computed, so you can SEE the effect of each stage rather
than just reading numbers.

Produces:
  1. before_after_cleaning.png - histograms of each affected column, raw
     (disguised zeros included) vs. cleaned (zeros converted to missing,
     then filled in). This is the "before and after" for data cleaning.
  2. correlation_heatmap.png - how strongly each measurement relates to
     every other measurement and to the diabetes outcome.
  3. class_balance.png - how many patients have diabetes vs. don't, in the
     raw data before any model sees it.
  4. roc_curve.png - the "before and after prediction" picture: how well
     each model separates diabetic from non-diabetic patients, compared to
     random guessing.
  5. confusion_matrices.png - for each model, how many predictions were
     right vs. wrong, broken down by type of error.
  6. feature_importance.png - which measurements each model leaned on most.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from data_prep import ZERO_AS_MISSING, load_raw, load_clean
from model import FEATURE_COLS, build_logistic_pipeline, build_rf_pipeline, split_data

FIG_DIR = Path("results/figures")


def plot_before_after_cleaning(raw: pd.DataFrame, clean: pd.DataFrame):
    fig, axes = plt.subplots(len(ZERO_AS_MISSING), 2, figsize=(10, 3 * len(ZERO_AS_MISSING)))
    for i, col in enumerate(ZERO_AS_MISSING):
        axes[i, 0].hist(raw[col], bins=30, color="#c0392b")
        axes[i, 0].set_title(f"{col} - BEFORE (raw, 0 = missing test)")
        axes[i, 0].set_ylabel("number of patients")

        axes[i, 1].hist(clean[col], bins=30, color="#2980b9")
        axes[i, 1].set_title(f"{col} - AFTER (0s fixed, gaps filled in)")

    fig.suptitle("Effect of data cleaning: before vs. after", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "before_after_cleaning.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(clean: pd.DataFrame):
    numeric_cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                     "insulin", "bmi", "diabetes_pedigree", "age", "diabetes"]
    corr = clean[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="correlation (-1 to 1)")
    ax.set_title("How each measurement relates to the others\n(the bottom row/right column shows the link to diabetes itself)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(clean: pd.DataFrame):
    counts = clean["diabetes"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["No diabetes", "Diabetes"], counts.values, color=["#2980b9", "#c0392b"])
    for bar, count in zip(bars, counts.values):
        pct = count / counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2, count + 5, f"{count}\n({pct:.0f}%)", ha="center")
    ax.set_title("How many patients have diabetes vs. don't\n(before any model touches the data)")
    ax.set_ylabel("number of patients")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance.png", dpi=120, bbox_inches="tight")
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
    ax.set_title("ROC curve: how well each model separates the two outcomes\n(closer to the top-left corner = better)")
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
    fig.suptitle("Confusion matrix: predicted vs. actual outcome on patients the model never saw during training", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return log_reg, rf


def plot_feature_importance(log_reg, rf):
    importances = rf.named_steps["clf"].feature_importances_
    coefs = log_reg.named_steps["clf"].coef_[0]

    order = np.argsort(importances)[::-1]
    features_sorted = [FEATURE_COLS[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(features_sorted[::-1], importances[order][::-1], color="#27ae60")
    axes[0].set_title("Random Forest: which measurements mattered most")
    axes[0].set_xlabel("importance")

    coef_order = np.argsort(np.abs(coefs))[::-1]
    coef_features_sorted = [FEATURE_COLS[i] for i in coef_order]
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in coefs[coef_order]]
    axes[1].barh(coef_features_sorted[::-1], coefs[coef_order][::-1], color=colors[::-1])
    axes[1].set_title("Logistic Regression: weight given to each measurement\n(red = raises risk, blue = lowers it)")
    axes[1].set_xlabel("coefficient")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    clean = load_clean()

    print("Plotting before/after cleaning...")
    plot_before_after_cleaning(raw, clean)

    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(clean)

    print("Plotting class balance...")
    plot_class_balance(clean)

    print("Training models for ROC / confusion matrix plots...")
    X_train, X_test, y_train, y_test = split_data(clean)
    log_reg, rf = plot_roc_and_confusion(X_train, X_test, y_train, y_test)

    print("Plotting feature importance...")
    plot_feature_importance(log_reg, rf)

    print(f"\nDone. 6 images saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

"""
Data understanding: how each measurement relates to the target (diabetes),
looked at one feature at a time, before any model gets built.

Run: python src/data_understanding.py
Writes:
  - results/figures/boxplots_by_outcome.png
  - results/figures/rate_by_bins.png

This is deliberately kept separate from modeling. The point of this stage
is to look at the raw relationships with our own eyes and form an
expectation, BEFORE letting a model tell us what matters. If the model's
later feature importance ranking matches what we see here, that's a good
sign the model learned something real rather than something spurious.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_prep import load_clean

FIG_DIR = Path("results/figures")

NUMERIC_FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]

# Why each feature is a biologically reasonable thing to use for predicting
# diabetes, stated up front rather than left implicit. This is printed to
# the terminal and also written into results/rationale.txt.
RATIONALE = {
    "pregnancies": "A history of gestational diabetes during pregnancy is a known "
                    "risk factor for developing type 2 diabetes later, so number "
                    "of pregnancies is a reasonable, if indirect, proxy.",
    "glucose": "Blood glucose is literally part of how diabetes is diagnosed. "
               "This is the most direct measurement in the dataset, not just a proxy.",
    "blood_pressure": "High blood pressure and diabetes frequently occur together "
                       "as part of what's clinically called metabolic syndrome, "
                       "so it's a plausible, if weaker, indirect signal.",
    "skin_thickness": "A proxy for body fat. Higher body fat is linked to insulin "
                       "resistance, which is the core mechanism behind type 2 diabetes.",
    "insulin": "Measures the body's own insulin response directly. Abnormal "
               "insulin levels are a direct sign of the insulin resistance "
               "that defines type 2 diabetes.",
    "bmi": "Obesity is one of the strongest and most well-established modifiable "
           "risk factors for type 2 diabetes.",
    "diabetes_pedigree": "A summary score of family history. Diabetes has a real "
                          "genetic component, so family history is a legitimate, "
                          "if imprecise, risk signal.",
    "age": "Risk of type 2 diabetes rises with age as insulin sensitivity "
           "naturally declines over time.",
}


def plot_boxplots_by_outcome(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, feature in zip(axes, NUMERIC_FEATURES):
        data_no = df.loc[df["diabetes"] == 0, feature].dropna()
        data_yes = df.loc[df["diabetes"] == 1, feature].dropna()
        ax.boxplot([data_no, data_yes], tick_labels=["No diabetes", "Diabetes"])
        ax.set_title(feature)
    fig.suptitle(
        "Each measurement, split by outcome\n"
        "(if the two boxes sit at clearly different heights, that feature likely helps the model)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplots_by_outcome.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rate_by_bins(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for ax, feature in zip(axes, NUMERIC_FEATURES):
        try:
            bins = pd.qcut(df[feature], q=4, duplicates="drop")
        except ValueError:
            bins = pd.cut(df[feature], bins=4)
        rate_by_bin = df.groupby(bins, observed=True)["diabetes"].mean()
        span = df[feature].max() - df[feature].min()
        decimals = 2 if span < 5 else 0
        labels = [f"{i.left:.{decimals}f}-{i.right:.{decimals}f}" for i in rate_by_bin.index]
        bars = ax.bar(range(len(rate_by_bin)), rate_by_bin.values, color="#8e44ad")
        ax.set_xticks(range(len(rate_by_bin)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(feature)
        ax.set_ylabel("diabetes rate")
        for bar, val in zip(bars, rate_by_bin.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.0%}",
                    ha="center", fontsize=7)
    fig.suptitle(
        "Diabetes rate across four equal-sized groups of each measurement\n"
        "(low to high, left to right, a rising trend means that feature carries real signal)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_by_bins.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    print("Plotting boxplots by outcome...")
    plot_boxplots_by_outcome(df)

    print("Plotting diabetes rate by feature bins...")
    plot_rate_by_bins(df)

    print("\nWhy each feature was used to predict the target:\n")
    lines = []
    for feature, reason in RATIONALE.items():
        line = f"- {feature}: {reason}"
        print(line)
        lines.append(line)

    Path("results").mkdir(exist_ok=True)
    with open("results/rationale.txt", "w") as f:
        f.write("Why each feature was used to predict diabetes\n")
        f.write("=" * 50 + "\n\n")
        f.write("\n".join(lines))

    print(f"\nDone. 2 images saved to {FIG_DIR}/, rationale saved to results/rationale.txt")

if __name__ == "__main__":
    main()

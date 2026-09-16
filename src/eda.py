"""
Exploratory analysis for the Pima Indians Diabetes dataset.

Run directly: python src/eda.py
Writes a plain-text summary to results/eda_summary.txt so findings are
captured even if you're skimming the repo rather than re-running it.
"""
from pathlib import Path

import pandas as pd

from data_prep import load_clean

NUMERIC_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]


def run_eda(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Shape: {df.shape}")
    lines.append("")
    lines.append("Summary statistics:")
    lines.append(df.describe().round(2).T.to_string())
    lines.append("")
    lines.append("Missing values per column (after zero -> NaN conversion):")
    lines.append(df.isna().sum().to_string())
    lines.append("")
    lines.append("Class balance (diabetes outcome):")
    lines.append(df["diabetes"].value_counts(normalize=True).round(3).to_string())
    lines.append("")
    lines.append("Correlation of numeric features with outcome:")
    corr = df[NUMERIC_COLS + ["diabetes"]].corr()["diabetes"].drop("diabetes")
    lines.append(corr.sort_values(ascending=False).round(3).to_string())
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_clean()
    summary = run_eda(df)
    print(summary)

    Path("results").mkdir(exist_ok=True)
    with open("results/eda_summary.txt", "w") as f:
        f.write(summary)

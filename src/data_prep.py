"""
Data loading and cleaning for the Pima Indians Diabetes dataset.

Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
Original file has no header row; columns are documented in README.md.

Key cleaning decision: several columns use 0 as a placeholder for a missing
measurement (a living patient cannot have 0 blood pressure or 0 BMI). We
convert those to NaN so downstream imputation treats them honestly, and add
missingness-indicator flags for the two worst-affected columns (insulin,
skin_thickness) in case the fact a test wasn't ordered is itself informative.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/pima_diabetes_raw.csv"

COLUMN_NAMES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "diabetes",
]

# Columns where 0 is not a physiologically valid value and really means "missing"
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and attach documented column names."""
    return pd.read_csv(path, names=COLUMN_NAMES)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convert disguised-zero missing values to NaN and add missingness flags."""
    df = df.copy()
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    df["insulin_missing"] = df["insulin"].isna().astype(int)
    df["skin_thickness_missing"] = df["skin_thickness"].isna().astype(int)
    return df


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    return clean(load_raw(path))


if __name__ == "__main__":
    df = load_clean()
    print(df.shape)
    print(df.isna().sum())

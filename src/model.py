"""
Model training and evaluation: logistic regression vs. random forest.

Run directly: python src/model.py
Writes results/model_report.txt and results/feature_importance.csv.

Why two models: logistic regression is the interpretable baseline; random
forest checks whether there's meaningful non-linear/interaction structure
the linear model misses. Comparing coefficient signs against feature
importances is a quick sanity check that both models agree on what matters.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_clean

FEATURE_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
    "bmi", "diabetes_pedigree", "age", "insulin_missing", "skin_thickness_missing",
]
TARGET_COL = "diabetes"
RANDOM_STATE = 42


def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    # Stratified split: preserves the ~35% positive rate in both sets
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)


def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def evaluate(name: str, model: Pipeline, X_test, y_test) -> str:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    lines = [
        f"=== {name} ===",
        f"AUC: {roc_auc_score(y_test, proba):.3f}",
        classification_report(y_test, pred, digits=3),
        f"Confusion matrix:\n{confusion_matrix(y_test, pred)}",
    ]
    return "\n".join(lines)


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = split_data(df)

    log_reg = build_logistic_pipeline().fit(X_train, y_train)
    rf = build_rf_pipeline().fit(X_train, y_train)

    report = []
    report.append(evaluate("Logistic Regression", log_reg, X_test, y_test))
    report.append(evaluate("Random Forest", rf, X_test, y_test))

    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "logistic_coef": log_reg.named_steps["clf"].coef_[0],
        "rf_importance": rf.named_steps["clf"].feature_importances_,
    }).sort_values("rf_importance", ascending=False)

    report.append("\n=== Feature comparison (sorted by RF importance) ===")
    report.append(coef_df.round(4).to_string(index=False))

    full_report = "\n\n".join(report)
    print(full_report)

    Path("results").mkdir(exist_ok=True)
    with open("results/model_report.txt", "w") as f:
        f.write(full_report)
    coef_df.to_csv("results/feature_importance.csv", index=False)


if __name__ == "__main__":
    main()

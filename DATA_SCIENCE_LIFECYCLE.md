# Data Science Lifecycle: Diabetes Risk Prediction

This document walks through the complete process behind this project, stage
by stage, following the standard 7-stage data science lifecycle. Each stage
explains what was done, why, and what came out of it. This follows the same
structure as the companion project's lifecycle document
(`hospital-readmission-analysis/DATA_SCIENCE_LIFECYCLE.md`), so the two can
be compared side by side stage for stage.

1. [Problem Definition & Business Understanding](#1-problem-definition--business-understanding)
2. [Data Collection & Sourcing](#2-data-collection--sourcing)
3. [Data Cleaning & Preparation](#3-data-cleaning--preparation)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis-eda)
5. [Modeling & Experimentation](#5-modeling--experimentation)
6. [Insights, Visualization & Storytelling](#6-insights-visualization--storytelling)
7. [Documentation & Deployment](#7-documentation--deployment)

---

## 1. Problem Definition & Business Understanding

**Clear objective.** Predict whether a patient will test positive for
diabetes, using 8 routine measurements that could be collected during a
standard doctor's visit, without needing a formal glucose tolerance lab
test up front. This is an early-screening problem: could cheap, already
-available measurements flag who should be prioritized for a real
diagnostic test?

**Success metrics, chosen up front.** Plain accuracy was deliberately not
used as the primary metric (explained in stage 5). Instead: AUC (ranking
quality across all thresholds), recall (share of real diabetes cases
caught), and precision (share of flagged patients who are actually
positive). Success here means a model that clearly beats random guessing
by a wide, honestly reported margin, with the reasoning behind every
decision documented well enough to be checked or challenged.

**Constraints acknowledged from the start.** This is a demonstration and
learning project, not a certified clinical tool. It uses one specific,
narrow population (see stage 2), and that limitation shapes every
downstream conclusion.

## 2. Data Collection & Sourcing

**Data identification.** The Pima Indians Diabetes dataset, originally
compiled by the National Institute of Diabetes and Digestive and Kidney
Diseases (NIDDK), a U.S. government research agency. Retrieved from a
public GitHub mirror of the dataset as a plain CSV file.

**Data ingestion.** Downloaded programmatically (`curl`) directly into
`data/pima_diabetes_raw.csv`. No manual entry, scraping, or private data
sources involved. This is a historical, static dataset, not a live feed.

**What's in it.** 768 patient records, all women aged 21 and older from
the Pima community, an Indigenous American population studied due to
unusually high diabetes prevalence. 8 measurements per patient plus the
diagnosis outcome (see the column dictionary in `README.md` for full
detail on each one).

**Collection caveats, carried forward honestly:**
- Single population, single sex. No claim of generalization to any other
  group is supported by this data.
- Small sample size (768 rows) by modern standards, meaning any metric
  reported later carries real uncertainty.
- Retrospective data, not collected specifically for this analysis. No
  control over what was or wasn't measured.

## 3. Data Cleaning & Preparation

**Handling missing values, the central cleaning problem here.** Five
columns (`glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `bmi`)
use `0` as a placeholder for a measurement that was never taken. A living
patient cannot have a blood pressure of zero, so this is disguised missing
data, not a real reading. Left uncaught, a model would treat these zeros
as real information and learn something false. `src/data_prep.py`
converts every one of these to a proper missing value, then fills the gap
with the median value for that column, a simple, robust choice given the
scale of the gaps (see the table below).

| Column | % disguised as zero |
|---|---|
| `insulin` | 48.7% |
| `skin_thickness` | 29.6% |
| `blood_pressure` | 4.6% |
| `bmi` | 1.4% |
| `glucose` | 0.7% |

**Removing duplicates and errors.** No duplicate patient records were
found in this dataset. Data types were already consistent (all numeric)
on load, so no type-correction was needed.

**Feature engineering.** No new features were manufactured, but the
missing-value pattern itself was checked for informativeness: `insulin`
and `skin_thickness` have the highest missingness, worth knowing about
even though we ultimately used simple imputation rather than a
missingness-flag approach for this project (unlike the readmission
project, where flags for missingness were tested and found to make little
difference; see that project's lifecycle document, stage 3).

**Visual proof the cleaning worked**, `src/visualize.py` produces
`results/figures/before_after_cleaning.png`, showing each affected
column's distribution before (disguised zeros included) and after
(cleaned and imputed). The zero-spike disappears visibly for `insulin` and
`skin_thickness` in particular.

## 4. Exploratory Data Analysis (EDA)

**Summary statistics and class balance** (`src/eda.py`): 768 patients,
34.9% tested positive for diabetes. Not a severe imbalance, so plain
accuracy would not have been wildly misleading here (unlike the
readmission project), though AUC, precision, and recall were still used
as the primary judgment, for consistency and because they answer the
actual question of interest more directly.

**Visual patterns** (`src/data_understanding.py`, run before any model was
built):
- Boxplots of each measurement split by outcome show `glucose` and `bmi`
  with the clearest separation between diabetic and non-diabetic patients;
  `blood_pressure` and `skin_thickness` show the least.
- Binning each measurement into four groups and tracking the diabetes rate
  across them shows `glucose` with by far the strongest relationship: 7%
  in the lowest quarter of readings, climbing to 70% in the highest.

**Early hypothesis formed from this stage:** glucose and BMI would likely
dominate any model built on this data, given the clarity of their
relationship with the outcome even before any modeling was attempted. This
hypothesis is checked directly in stage 5.

**A rationale was written for every feature** before modeling
(`results/rationale.txt`), stating in one sentence why each measurement is
a biologically reasonable predictor, not just a statistically correlated
one. For example: glucose is used because it is literally part of how
diabetes is diagnosed clinically, not merely correlated with it.

## 5. Modeling & Experimentation

**Algorithm selection.** Two structurally different models, chosen
deliberately for comparison rather than to chase the single best score:

1. **Logistic Regression**, an interpretable linear baseline. Assigns a
   weight to each measurement and sums them into a risk score.
2. **Random Forest**, an ensemble of decision trees, capable of capturing
   non-linear patterns the linear model might miss.

**Training and tuning.** Data was split 75/25 into training and test sets,
stratified to preserve the 34.9% positive rate in both. Numeric features
were median-imputed then standardized; the Random Forest was deliberately
kept shallow (`max_depth=6`, `min_samples_leaf=20`) to avoid overfitting
noise given the small sample size, a conscious trade-off favoring a fair
comparison over squeezing out maximum training-set performance.

**Model evaluation, on the held-out test set:**

| Model | AUC | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.83 | 52% | 60% |
| Random Forest | 0.82 | 48% | 64% |

Logistic regression edged out the Random Forest slightly, which is
expected rather than surprising: the underlying relationship in this data
is close to additive (each risk factor contributes somewhat independently),
which is exactly what logistic regression is built to capture. This is a
useful general lesson: more model flexibility isn't automatically better,
it depends on whether the extra flexibility is buying you anything given
how the outcome actually arises.

## 6. Insights, Visualization & Storytelling

**Actionable takeaways.** Glucose dominates every other measurement (a
finding confirmed independently by both models, not just one), followed by
BMI and age. This matches the early hypothesis formed during EDA (stage 4),
which is itself a useful check: when a model's conclusions agree with what
was visible in the raw data before any modeling, that's a sign the model
learned something real.

**Visual reports** (`results/figures/`, from `src/visualize.py`):
- `roc_curve.png`: both models' ranking ability plotted against random
  guessing across every threshold, not just the default cutoff.
- `confusion_matrices.png`: the literal count of correct catches, missed
  cases, and false alarms behind the precision/recall numbers above.
- `feature_importance.png`: side-by-side ranking from both models, used to
  confirm agreement on what matters rather than trusting one model's
  opinion alone.
- `correlation_heatmap.png` and `class_balance.png`: earlier-stage context
  carried through into the final report.

**Translating for a non-technical reader.** All of the above is written up
in plain language, with every statistical and medical term explained on
first use, in `README.md`, specifically so someone without a data science
or medical background can follow and independently check the reasoning.

## 7. Documentation & Deployment

**Code repository.** The full project (raw data, all source code,
generated results, and this document) is version-controlled with git and
structured for GitHub: a plain README for practical use, this document for
the full formal record, and clearly separated `src/` and `data/` folders.

**Reproducibility.** `requirements.txt` pins every dependency. The full
pipeline runs end to end with five commands (see `README.md`, "How to run
this yourself"), each writing its output to `results/` so a fresh clone
can be verified without needing to trust the numbers on faith.

**Deployment status: not deployed, deliberately.** This project stops at a
tested, documented model, not a live prediction service. See
`README.md`'s "Recommendations for practitioners" section for what a
responsible real-world use would require before that step (threshold
tuning against real costs, validation on a broader population,
periodic re-evaluation), and the "Limitations" section for what stands in
the way of deployment as-is.

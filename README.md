# Predicting Loan Default Using Machine Learning
## The problem

Banks lose money when borrowers stop repaying (**default**). Given the information on a loan application, can a
model estimate the chance an applicant will default *before* the loan is approved? This is a binary
**classification** task: `TARGET = 1` (default / payment difficulties) vs `TARGET = 0` (repaid).

## Dataset

| | |
|---|---|
| Source | **Home Credit Default Risk** — public Kaggle competition (2018) |
| File used for training | `application_train.csv` — 307,511 applicants × 122 columns (~166 MB) |
| Also read | `application_test.csv` — 48,744 applicants, no `TARGET` (Kaggle's hidden test set) |
| Positive class | 24,825 defaults (**8.1%**) |
| Access | The notebook downloads both files automatically from the original repo's Git LFS storage. Fallback: Kaggle API (instructions inside the notebook). |

![Class imbalance](images/r1_target_imbalance.png)

## Pipeline (as in the original notebooks)

![Pipeline](images/r5_pipeline.png)

1. **EDA** — distributions of contract type, gender, car/realty ownership, income, credit amount, education; default vs non-default comparisons.
2. **Missing values** — drop columns > 50% missing (122 → 81), mean-fill 6 building columns, keep columns < 10% missing, mean-fill numeric and mode-fill categorical blanks; drop the 4 `CODE_GENDER = 'XNA'` rows.
3. **Split & scale** — 70/30 `train_test_split(random_state=101)`, `StandardScaler`.
4. **Class balancing** — `RandomOverSampler`, `SMOTE` (also wired: `TomekLinks`, `NearMiss`).
5. **Models** — Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest; `GridSearchCV` over `n_estimators`.
6. **Metrics** — accuracy, precision, recall, F1, classification report, confusion matrix, ROC-AUC.

## Key EDA findings

![Missing values](images/r2_missing_top.png)

41 of 122 columns are more than half empty — mostly detailed descriptions of the applicant's building.

![Default rate by group](images/r4_default_rate_by_group.png)

![External score](images/u17_ext_source.png)

- Defaults are rare (8.1%), so accuracy alone is misleading — always predicting "repay" scores 91.9%.
- The external credit scores (`EXT_SOURCE_*`) are by far the strongest single signal: the lowest-scored tenth defaults ~6× as often as the highest.
- Younger applicants, those with lower-secondary education (10.9% vs 5.4% for higher education) and "Working" income types (9.6% vs 5.4% for pensioners) default more often.
- Income on its own separates the classes poorly.

## Results reproduced from the original notebooks

Left number: the output saved in the original repo's notebook. Right number: a clean top-to-bottom run of this notebook.

| Model | Balancing | ROC-AUC: original repo → this notebook |
|---|---|---|
| Logistic Regression | RandomOverSampler | 0.533 → 0.699 |
| Gaussian Naive Bayes | RandomOverSampler | 0.515 → 0.644 |
| Decision Tree | RandomOverSampler | 0.565 → 0.523 |
| Random Forest | RandomOverSampler | 0.562 → 0.683 |
| Decision Tree | SMOTE | 0.804 → 0.794 |
| **Random Forest** | **SMOTE** | **0.947 → 0.948** |

The SMOTE rows reproduce closely (SMOTE is unseeded, so they wobble slightly). The RandomOverSampler rows differ
because the original outputs came from an out-of-order session: the saved output of the resampling cell reads
`Original dataset shape: (395864,)`, i.e. it resampled data that had *already* been resampled. A clean in-order run
gives the right-hand numbers.

## Why the headline score is inflated

The original code resamples the **validation** set as well as the training set
(`X_cv, y_cv = sampler_function(X_cv_standardized, y_cv, ...)`). With SMOTE that means the model is scored mostly on
synthetic points interpolated between real defaulters, at a 50/50 class ratio — not on real applicants. Two
smaller issues: the row ID `SK_ID_CURR` is used as a feature, and in the "categorical + numerical" section
`pd.DataFrame(...).reset_index(drop=True, inplace=True)` returns `None`, so the one-hot columns are silently never
added (both sections print identical reports).

![Original vs honest](images/r7_original_vs_honest.png)

The original cells are kept unchanged so the published results can be reproduced; the fix is in a separate section.

## Leakage-free re-evaluation (Bonus section of the notebook)

SMOTE applied to training data only, ID column dropped, one-hot categories actually included, scored on 92,253
untouched validation applicants:

| Model | ROC-AUC on real validation data |
|---|---|
| Logistic Regression + SMOTE (train only) | 0.700 |
| Logistic Regression, `class_weight='balanced'` | 0.708 |
| Random Forest, `min_samples_leaf=50` | 0.709 |

![Honest ROC](images/r8_honest_roc.png)

![Feature importance](images/r9_feature_importance.png)

For context, top Kaggle solutions — which joined all seven Home Credit tables and used gradient boosting — scored
about 0.80 AUC.

---

## What changed vs the original code

Only what was needed for current pandas / scikit-learn; every changed line is tagged `# [Colab fix]` in the notebook.

| Change | Why |
|---|---|
| `plot_confusion_matrix` → small shim around `ConfusionMatrixDisplay.from_estimator` | removed in scikit-learn 1.2 |
| `Series[0]` → `Series.iloc[0]` | positional `[]` on a label index removed in pandas 3 |
| chained `df[col].fillna(..., inplace=True)` → `df[col] = df[col].fillna(...)` | no effect under pandas Copy-on-Write; would silently change which columns survive |
| `sns.kdeplot(x='amt_income_total')` → `'AMT_INCOME_TOTAL'` in EDA cell 7 | the column wasn't lower-cased yet (KeyError) |
| `pd.set_option("future.infer_string", False)` | keep text columns as `object` dtype, as the original expects |
| Added: pip install, data download, memory cleanup between parts, Bonus section | Colab convenience / honest evaluation |

## Files

```
Loan_Default_Prediction_Colab.ipynb   the notebook (Parts 1–3 = original, Bonus = added); saved with outputs from a full run
README.md                             this file
understand.md                         beginner's guide with explanatory plots
requirements.txt                      versions it was tested with
images/                               all figures used in the two markdown files
```

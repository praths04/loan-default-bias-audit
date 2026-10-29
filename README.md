# Auditing a Loan-Default Model for Bias

A capstone hackathon project (solo, Python) that predicts loan-default risk while treating explainability and fairness as first-class requirements. Protected attributes (gender, age, education) were named before modeling, then checked to see whether the model leaned on them.

## What the notebook does

1. **Data preparation:** 5,859 training records and 122 raw columns. Dropped 44 columns with more than 50% missing values, fixed an employment-days anomaly, and engineered derived features.
2. **Feature selection:** Random Forest feature importance narrowed 81 candidate features to 62 predictive features.
3. **Class imbalance:** about 7.8% of loans default. Class weights were used as the primary approach (SMOTE was compared) to avoid amplifying existing bias.
4. **Modeling:** five tuned model families (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM). The best, Logistic Regression, reached a validation ROC-AUC of 0.78 with 76% recall.
5. **Responsible-AI audit:** SHAP, Fairlearn and disparate-impact checks. Gender (0.77), age (0.45) and education (0.59) all failed the standard 80% threshold.
6. **Mitigation:** removing the 4 protected attributes raised gender's disparate-impact ratio from 0.77 to 0.96 with ROC-AUC nearly unchanged (0.78 to 0.77). Education improved to 0.63 and age did not improve, so both still need monitoring.

## Data

The data was sourced from Kaggle. The files (`train_x3.csv` and `test_x3 (1).csv`) are not included in this repository.

## Running it

The notebook was written in Google Colab and reads the two CSV files above. Download the data from Kaggle, place the files in the same folder as the notebook, and update the file paths in the first cells.

## Tools

Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, LIME, DiCE, Fairlearn.

# Payment Transaction Anomaly Classification

## Project Overview

Payment Transaction Anomaly Classification is a machine learning project
that classifies payment transactions into two categories:

- Routine / Legitimate Transaction
- Anomalous / Fraudulent Transaction

The project follows an end-to-end machine learning workflow including
data investigation, exploratory data analysis, preprocessing, model
development, evaluation, error analysis, and deployment using Streamlit.

---

## Problem Statement

The objective of this project is to develop a machine learning solution
that can classify transaction records as legitimate or fraudulent based
on the available transaction characteristics.

The system also provides a simple application for demonstrating
predictions on transaction data.

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset obtained
from Kaggle.

The dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

### Target Variable

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The `V1` to `V28` features are anonymized numerical features provided
by the dataset.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Machine Learning Workflow

The project includes the following steps:

1. Dataset loading
2. Data quality investigation
3. Missing-value analysis
4. Duplicate analysis
5. Class-balance analysis
6. Exploratory Data Analysis
7. Outlier investigation
8. Data leakage checking
9. Train-test splitting
10. Feature scaling
11. Logistic Regression
12. Decision Tree
13. Model evaluation
14. Error analysis
15. Final model selection
16. Model saving
17. Streamlit application development

---

## Models Used

### Logistic Regression

Logistic Regression was used as a baseline classification model.

### Decision Tree

A Decision Tree classifier was also trained with a maximum depth of 10.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

---

## Model Results

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 99.79% |
| Precision | 70.00% |
| Recall | 77.78% |
| F1-score | 73.68% |
| ROC-AUC | 0.9922 |

### Decision Tree

| Metric | Score |
|---|---:|
| Accuracy | 99.83% |
| Precision | 72.73% |
| Recall | 88.89% |
| F1-score | 80.00% |
| ROC-AUC | 0.9438 |

The Decision Tree was selected as the final model for the prototype
because it achieved higher recall and F1-score on the test set.

Logistic Regression achieved a higher ROC-AUC, which is also considered
when interpreting the results.

---

## Error Analysis

False positives and false negatives were analyzed using the test-set
predictions.

- **False Positive:** A legitimate transaction incorrectly classified
  as fraudulent.
- **False Negative:** A fraudulent transaction incorrectly classified
  as legitimate.

False-negative errors are particularly important in fraud detection
because fraudulent transactions may be missed.

---

## Streamlit Application

A Streamlit application was developed to demonstrate the trained model.

The application supports:

- Manual transaction input
- Real legitimate transaction testing
- Real fraudulent transaction testing
- Input validation
- Feature scaling
- Model prediction
- Clear prediction results

The application displays either:

**Routine / Legitimate Transaction**

or

**Anomalous / Fraudulent Transaction**

---

## Project Structure

```text
Payment-Transaction-Anomaly/
│
├── app/
├── data/
│   └── archive (1)/
│       └── creditcard.csv
├── models/
├── notebooks/
├── paper/
│
├── decision_tree_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── app.py
├── requirements.txt
└── README.md
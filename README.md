# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)

---

## Project Description

This project performs a churn analysis based on bank data and trains two models to predict customer churns.
The script performs the following tasks:
- EDA analysis
- Train Test split
- Training of a Random Forest model
- Training of a Logistic Regression model
- Generation of classification reports for the trained models

---

## Files and Data Description

The project uses three main files and a data file.

### Main Files

- `churn_library.py`  
  Main script to perform churn analyis and model training.

- `churn_script_logging_and_tests.py`  
  Script to run tests and log results for `churn_library.py`.

- `churn_notebook.ipynb`  
  Basis for the implementation of `churn_library.py`.

---

### Data

- `data/bank_data.csv`  
  Bank data for churn analysis
  - target: churn (derived from attrition flag)
  - features: e.g. total trans amount, marital status, income group

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

---

## Running the Files

### 1. Run the Pipeline

```bash
python churn_library.py
```

- Runs EDA analysis and stores images
- Performs train test split
- Trains Random Forest and Logistic Regression model
- Stores best models
- Generates and stores model result reports

---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py
```

- Runs tests for `churn_library.py`.
- Test results are stored in a log file.

---

## Expected Outputs

- Models:
  - `models/rfc_model.pkl`
  - `models/logistic_model.pkl`

- Images:
  - `images/eda/churn_distribution.png`
  - `images/eda/correlation_heatmap.png`
  - `images/eda/customer_age_distribution.png`
  - `images/eda/marital_status_distribution.png`
  - `images/eda/total_trans_ct_distribution.png`

  - `images/results/classification_report_lr.png`
  - `images/results/classification_report_rf.png`
  - `images/results/feature_importance.png`
  - `images/results/rov_curve.png`

- Logs:
  - `logs/churn_library.log`

---

## Notes

- Target feature churn is derived during EDA analysis

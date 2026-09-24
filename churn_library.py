"""
Script for performing churn analysis and model training.

Author: Thomas
Date created: 2026-09-17
"""
# DONE:
# Add a module-level docstring describing:
# - Purpose of this file
# - Author
# - Date created

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# DONE: add required imports
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    # DONE: implement
    # Hint:
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    # DONE: implement
    # Suggested steps:
    # 1. Create a binary churn column if needed
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # 2. Plot key distributions
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig(os.path.join(EDA_DIR, 'churn_distribution.png'))
    plt.close()

    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(EDA_DIR, 'customer_age_distribution.png'))
    plt.close()

    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(EDA_DIR, 'marital_status_distribution.png'))
    plt.close()

    plt.figure(figsize=(20,10)) 
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(EDA_DIR, 'total_trans_ct_distribution.png'))
    plt.close()

    # 3. Plot a correlation heatmap
    plt.figure(figsize=(20, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(EDA_DIR, 'correlation_heatmap.png'))
    plt.close()

    # 4. Save figures into EDA_DIR
    # Removed redundant saving of figures since they are already saved above


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """
    # DONE: implement
    for category in category_lst:
        # Create a new column for the encoded feature
        new_col_name = f"{category}_{response}"
        # Calculate the mean of the response for each category
        category_means = df.groupby(category)[response].mean()
        # Map the means to the original dataframe
        df[new_col_name] = df[category].map(category_means)

    return df


def perform_feature_engineering(df, response, category_cols, quant_cols):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
              category_cols: list of categorical columns
              quant_cols: list of quantitative columns
    output:
              x_train, x_test, y_train, y_test
    """
    # DONE: implement
    y = df[response]

    X = pd.DataFrame()
    encoded_cols = [f"{col}_{response}" for col in category_cols]
    keep_cols = quant_cols + encoded_cols
    X[keep_cols] = df[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(
    model_name,
    y_train,
    y_test,
    y_train_preds,
    y_test_preds,
    output_pth
):
    """
    Save classification reports as images.

    input:
            predictions and labels
    output:
            None
    """
    create_output_directories()

    # DONE: implement
    # Classification report for Random Forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.0, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.5, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.1, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(output_pth)
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    Save feature importance plot.

    input:
            model, x_data, output path
    output:
            None
    """
    # DONE: implement
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 14))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=45, ha='right')
    plt.savefig(output_pth)
    plt.close()


def roc_curve_plot(model_rf, model_lr, x_test, y_test, output_pth):
    """
    Save ROC curve plot.

    input:
            model_rf, model_lr, x_test, y_test, output path
    output:
            None
    """
    lrc_plot = RocCurveDisplay.from_estimator(model_lr, x_test, y_test)
    rfc_plot = RocCurveDisplay.from_estimator(model_rf, x_test, y_test)
    plt.figure(figsize=(15, 8))
    rfc_plot.plot(ax=plt.gca(), name='Random Forest')
    lrc_plot.plot(ax=plt.gca(), name='Logistic Regression')
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            train/test data
    output:
            None
    """
    create_output_directories()

    # TODO: implement
    # Setup Random Forest Classifier and hyperparameter grid for RandomizedSearchCV
    rfc = RandomForestClassifier(random_state=42)
    
    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini']
    }

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )

    # Setup Logistic Regression pipeline
    lrc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=3000))
    ])

    # Train models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, os.path.join(MODELS_DIR, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(MODELS_DIR, 'logistic_model.pkl'))

    # Calculate predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Print scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    # Save classification reports as images
    classification_report_image(
        'Random Forest',
        y_train, y_test,
        y_train_preds_rf, y_test_preds_rf,
        os.path.join(RESULTS_DIR, 'classification_report_rf.png')
    )
    classification_report_image(
        'Logistic Regression',
        y_train, y_test,
        y_train_preds_lr, y_test_preds_lr,
        os.path.join(RESULTS_DIR, 'classification_report_lr.png')
    )

    # Save feature importance plot
    feature_importance_plot(cv_rfc, x_train, os.path.join(RESULTS_DIR, 'feature_importance.png'))

    # Save ROC curve plot
    roc_curve_plot(cv_rfc, lrc, x_test, y_test, os.path.join(RESULTS_DIR, 'roc_curve.png'))


if __name__ == "__main__":
    create_output_directories()

    df = import_data(DATA_PATH)

    perform_eda(df)

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    quant_columns = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio'
    ]

    df = encoder_helper(df, category_columns, "Churn")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, "Churn", category_columns, quant_columns)
    train_models(x_train, x_test, y_train, y_test)

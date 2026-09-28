"""
Script for testing churn analysis and model training.

Author: Thomas
Date created: 2026-09-28
"""

import logging
import os

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

EDA_DIR = "./images/eda"
EDA_CHURN_DIST = os.path.join(EDA_DIR, "churn_distribution.png")
EDA_AGE_DIST = os.path.join(EDA_DIR, "customer_age_distribution.png")
EDA_MARITAL_DIST = os.path.join(EDA_DIR, "marital_status_distribution.png")
EDA_TOTAL_TRANS_CT_DIST = os.path.join(EDA_DIR, "total_trans_ct_distribution.png")
EDA_CORRELATION = os.path.join(EDA_DIR, "correlation_heatmap.png")

RESULTS_DIR = "./images/results"
RESULTS_CLASSIFICATION_REPORT_RF = os.path.join(RESULTS_DIR, "classification_report_rf.png")
RESULTS_CLASSIFICATION_REPORT_LR = os.path.join(RESULTS_DIR, "classification_report_lr.png")
RESULTS_FEATURE_IMPORTANCE = os.path.join(RESULTS_DIR, "feature_importance.png")
RESULTS_ROC_CURVE = os.path.join(RESULTS_DIR, "roc_curve.png")

MODELS_DIR = "./models"
MODEL_RFC = os.path.join(MODELS_DIR, "rfc_model.pkl")
MODEL_LR = os.path.join(MODELS_DIR, "logistic_model.pkl")

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

# configure logging to write INFO and ERROR messages
# to a .log file inside the ./logs directory
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    format='%(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)

        # add logging for success
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:

        # add logging for file not found
        logging.error("Testing import_data: FILE NOT FOUND")

        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:

        # add logging for empty dataframe
        logging.error("Testing import_data: THE DATAFRAME IS EMPTY")

        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        # assert output files exist
        assert os.path.exists(EDA_CHURN_DIST)
        assert os.path.exists(EDA_AGE_DIST)
        assert os.path.exists(EDA_MARITAL_DIST)
        assert os.path.exists(EDA_TOTAL_TRANS_CT_DIST)
        assert os.path.exists(EDA_CORRELATION)

        # logging success
        logging.info("Testing perform_eda: SUCCESS")

    except Exception as err:

        # logging failure
        logging.error("Testing perform_eda: FAILURE")

        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)

        # create response column
        cls.perform_eda(df)  # Assuming perform_eda adds the response column
        response = "Churn"

        # call encoder_helper
        df = encoder_helper(df, category_columns, response)

        # assert encoded columns exist
        for category in category_columns:
            encoded_col = f"{category}_{response}"
            assert encoded_col in df.columns

        # logging success
        logging.info("Testing encoder_helper: SUCCESS")

    except Exception as err:

        # logging failure
        logging.error("Testing encoder_helper: FAILURE")

        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)

        # prepare data
        cls.perform_eda(df)
        df = cls.encoder_helper(df, category_columns, "Churn")

        # call function
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn", category_columns, quant_columns)

        # assert outputs
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        # logging success
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except Exception as err:

        # logging failure
        logging.error("Testing perform_feature_engineering: FAILURE")

        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)

        # prepare data
        cls.perform_eda(df)
        df = cls.encoder_helper(df, category_columns, "Churn")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn", category_columns, quant_columns)

        # call train_models
        train_models(X_train, X_test, y_train, y_test)

        # assert model files + images exist
        assert os.path.exists(MODEL_RFC)
        assert os.path.exists(MODEL_LR)
        assert os.path.exists(RESULTS_CLASSIFICATION_REPORT_RF)
        assert os.path.exists(RESULTS_CLASSIFICATION_REPORT_LR)
        assert os.path.exists(RESULTS_FEATURE_IMPORTANCE)
        assert os.path.exists(RESULTS_ROC_CURVE)

        # logging success
        logging.info("Testing train_models: SUCCESS")

    except Exception as err:

        # logging failure
        logging.error("Testing train_models: FAILURE")

        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")

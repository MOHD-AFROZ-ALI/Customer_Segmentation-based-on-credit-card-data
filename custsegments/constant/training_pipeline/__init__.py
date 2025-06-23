# This file marks the directory as a Python package
import os
from datetime import datetime

# General constants
ARTIFACTS_DIR: str = "C:\\Users\\hp\\Desktop\\CUSTO\\customer_segments_productionjules\\artifacts"
LOG_DIR: str = "C:\\Users\\hp\\Desktop\\CUSTO\\customer_segments_productionjules\\logs" # As defined in logger.py
PIPELINE_NAME: str = "customer_segments_pipeline"
PIPELINE_ROOT: str = os.path.join(ARTIFACTS_DIR, PIPELINE_NAME)
TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Data Ingestion Constants ---
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_ROOT_DIR: str = os.path.join(PIPELINE_ROOT, DATA_INGESTION_DIR_NAME)
# Assuming the CSV is directly in customer_data relative to the new project root `customer_segments`
# The path will be `customer_segments/customer_data/credit_card_data.csv`
# For components, we might pass `customer_data/credit_card_data.csv` if CWD is `customer_segments`
RAW_DATA_DIR: str = "customer_data" # Relative to customer_segments/
DATA_FILE_NAME: str = "credit_card_data.csv" # Original dataset name
INGESTED_DATA_DIR: str = os.path.join(DATA_INGESTION_ROOT_DIR, "ingested_data")
INGESTED_TRAIN_FILE: str = "train.csv" # If splitting, otherwise same as DATA_FILE_NAME
# For this project, we'll use the full dataset for unsupervised learning
INGESTED_DATA_FILE: str = "full_data.csv"


# --- Data Validation Constants ---
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_ROOT_DIR: str = os.path.join(PIPELINE_ROOT, DATA_VALIDATION_DIR_NAME)
DATA_VALIDATION_STATUS_FILE: str = "validation_status.txt"
DATA_VALIDATION_REPORT_FILE: str = "validation_report.json" # Or .txt

# Define schema for validation (example, update with actual columns and types)
# Based on CSV inspection.
EXPECTED_SCHEMA: dict = {
    "CUST_ID": "object",
    "BALANCE": "float64",
    "BALANCE_FREQUENCY": "float64",
    "PURCHASES": "float64",
    "ONEOFF_PURCHASES": "float64",
    "INSTALLMENTS_PURCHASES": "float64",
    "CASH_ADVANCE": "float64",
    "PURCHASES_FREQUENCY": "float64",
    "ONEOFF_PURCHASES_FREQUENCY": "float64",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "float64",
    "CASH_ADVANCE_FREQUENCY": "float64",
    "CASH_ADVANCE_TRX": "int64", # Assuming conversion to int is okay after handling potential floats/NaNs
    "PURCHASES_TRX": "int64", # Assuming conversion to int is okay
    "CREDIT_LIMIT": "float64", # Can have NaNs
    "PAYMENTS": "float64",
    "MINIMUM_PAYMENTS": "float64", # Can have NaNs
    "PRC_FULL_PAYMENT": "float64",
    "TENURE": "int64" # Assuming conversion to int is okay
}
# List of required files/columns (can be just the main data file for now)
ALL_REQUIRED_FILES: list = [INGESTED_DATA_FILE] # If using the output of data ingestion


# --- Data Transformation Constants ---
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_ROOT_DIR: str = os.path.join(PIPELINE_ROOT, DATA_TRANSFORMATION_DIR_NAME)
PREPROCESSOR_OBJECT_FILE_NAME: str = "preprocessor.pkl"
TRANSFORMED_DATA_DIR: str = os.path.join(DATA_TRANSFORMATION_ROOT_DIR, "transformed_data")
TRANSFORMED_TRAIN_FILE: str = "train_transformed.csv" # If applicable
TRANSFORMED_DATA_FILE: str = "data_transformed.csv"


# --- Model Trainer Constants ---
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_ROOT_DIR: str = os.path.join(PIPELINE_ROOT, MODEL_TRAINER_DIR_NAME)
MODEL_OBJECT_FILE_NAME: str = "model.pkl" # As per requirement for the primary model
KMEANS_MODEL_NAME: str = "kmeans_model.pkl"
DBSCAN_MODEL_NAME: str = "dbscan_model.pkl" # If saving DBSCAN model separately

# Default Hyperparameters (can be tuned or set via config)
KMEANS_N_CLUSTERS: int = 3 # Example, will be determined by Elbow method
DBSCAN_EPS: float = 0.5    # Example, needs tuning
DBSCAN_MIN_SAMPLES: int = 5 # Example, needs tuning


# --- Model Evaluation Constants ---
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_ROOT_DIR: str = os.path.join(PIPELINE_ROOT, MODEL_EVALUATION_DIR_NAME)
MODEL_EVALUATION_METRICS_FILE: str = "evaluation_metrics.json"
MODEL_EVALUATION_INSIGHTS_FILE: str = "cluster_insights.txt" # For textual business insights
MODEL_EVALUATION_PLOTS_DIR: str = os.path.join(MODEL_EVALUATION_ROOT_DIR, "plots")

# --- Model Evaluation Constants ---
CLUSTERING_EVALUATION_DIR_NAME: str = "clustering_evaluation"


# --- EDA Analysis Constants ---
EDA_ANALYSIS_DIR_NAME: str = "eda_analysis"
EDA_ANALYSIS_ROOT_DIR: str = os.path.join(PIPELINE_ROOT, EDA_ANALYSIS_DIR_NAME)
EDA_PLOTS_DIR: str = os.path.join(EDA_ANALYSIS_ROOT_DIR, "plots")
EDA_REPORT_FILE: str = os.path.join(EDA_ANALYSIS_ROOT_DIR, "eda_report.txt")


# --- Final Model and Prediction Output (as per top-level structure) ---
FINAL_MODEL_DIR: str = "final_model" # Relative to customer_segments/
PREDICTION_OUTPUT_DIR: str = "prediction_output" # Relative to customer_segments/
BATCH_PREDICTION_OUTPUT_FILE: str = "clustered_results.csv"

# Target column for clustering (not applicable in unsupervised, but good to note if one existed for supervised)
# TARGET_COLUMN: str = "TARGET_COLUMN_NAME"

# Columns to drop (example, update after EDA)
COLUMNS_TO_DROP: list = ["CUST_ID"] # CUST_ID is an identifier and not useful for clustering
NUMERICAL_COLUMNS: list = [ # Updated based on CSV inspection, excluding CUST_ID
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
]
CATEGORICAL_COLUMNS: list = [] # No obvious categorical columns from the provided data head.

# --- Project Root Definition ---
# Assuming this file is at: customer_segments/custsegments/constant/training_pipeline/__init__.py
# PROJECT_ROOT should point to 'customer_segments'
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# --- Directories relative to PROJECT_ROOT_DIR ---
# These are the top-level directories as per the desired structure
FINAL_MODEL_PARENT_DIR: str = os.path.join(PROJECT_ROOT_DIR, "final_model")
PREDICTION_OUTPUT_PARENT_DIR: str = os.path.join(PROJECT_ROOT_DIR, "prediction_output")
CUSTOMER_DATA_PARENT_DIR: str = os.path.join(PROJECT_ROOT_DIR, "customer_data") # Storing the raw data

# Ensure PIPELINE_ROOT and LOG_DIR are created (typically done by a setup script or main runner)
# For components, they will create their own specific subdirectories under ARTIFACTS_DIR
# os.makedirs(PIPELINE_ROOT, exist_ok=True) # This is good
# os.makedirs(LOG_DIR, exist_ok=True) # This is also good, but logger.py already does this.

# Ensure top-level output directories exist (can also be done by pipeline runner or app setup)
# os.makedirs(FINAL_MODEL_PARENT_DIR, exist_ok=True)
# os.makedirs(PREDICTION_OUTPUT_PARENT_DIR, exist_ok=True)
# os.makedirs(CUSTOMER_DATA_PARENT_DIR, exist_ok=True) # This is where the input data will be placed

# The `EXPECTED_SCHEMA` above is based on common column names for this dataset type.
# It's crucial to verify this against the actual dataset.
# `NUMERICAL_COLUMNS` and `CATEGORICAL_COLUMNS` also depend on this.
# `COLUMNS_TO_DROP` is also an initial guess.
# These will be refined in the Data Validation and EDA stages.

# --- Schema and Column Definitions (to be verified with actual data) ---
# (EXPECTED_SCHEMA, COLUMNS_TO_DROP, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS remain the same for now)
# It's important to load and inspect the CSV to finalize these.
# For example, 'CUST_ID' should definitely be dropped before training.
# 'MINIMUM_PAYMENTS' and 'CREDIT_LIMIT' might have NaNs that need careful handling.
# It's crucial to verify this against the actual dataset.
# `NUMERICAL_COLUMNS` and `CATEGORICAL_COLUMNS` also depend on this.
# `COLUMNS_TO_DROP` is also an initial guess.
# These will be refined in the Data Validation and EDA stages.

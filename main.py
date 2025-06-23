import sys
import traceback

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.pipeline.training_pipeline import TrainingPipeline
from custsegments.constant.training_pipeline import TIMESTAMP, CUSTOMER_DATA_PARENT_DIR, PROJECT_ROOT_DIR, DATA_FILE_NAME

# For ensuring data exists for the pipeline run (can be part of pre-run checks)
from pathlib import Path
import os
import pandas as pd
import numpy as np
from custsegments.constant.training_pipeline import NUMERICAL_COLUMNS


def setup_initial_data_if_needed():
    """
    Checks if the primary data file exists in customer_data.
    If not, tries to copy it from the original repository location (1) Dataset/).
    If still not found, creates a minimal dummy CSV to allow the pipeline to run.
    This is primarily for ease of testing and initial setup.
    """
    target_customer_data_path = Path(CUSTOMER_DATA_PARENT_DIR) / DATA_FILE_NAME
    original_data_path = Path(PROJECT_ROOT_DIR) / "customer_data" / "credit_card_data.csv" # Path from original repo structure

    if not target_customer_data_path.exists():
        logging.info(f"{target_customer_data_path} not found.")
        if original_data_path.exists():
            try:
                os.makedirs(target_customer_data_path.parent, exist_ok=True)
                import shutil
                shutil.copy(original_data_path, target_customer_data_path)
                logging.info(f"Copied data from {original_data_path} to {target_customer_data_path}.")
            except Exception as e:
                logging.warning(f"Could not copy data from {original_data_path} to {target_customer_data_path}: {e}. Will attempt to create dummy data.")
                create_dummy_data(target_customer_data_path)
        else:
            logging.warning(f"Original data also not found at {original_data_path}. Creating dummy data at {target_customer_data_path}.")
            create_dummy_data(target_customer_data_path)
    else:
        logging.info(f"Data file already exists at {target_customer_data_path}.")

def create_dummy_data(file_path: Path):
    """Creates a minimal dummy CSV file."""
    try:
        os.makedirs(file_path.parent, exist_ok=True)
        cols_for_dummy = ["CUST_ID"] + NUMERICAL_COLUMNS
        # Ensure NUMERICAL_COLUMNS is not empty
        if not NUMERICAL_COLUMNS:
             logging.error("NUMERICAL_COLUMNS is empty, cannot create dummy data without feature names.")
             return # Or raise error

        # Create a few rows of dummy data
        dummy_data_rows = []
        for i in range(5): # Create 5 rows
            row = [f"DUMMY_CUST{i+1}"]
            for _ in NUMERICAL_COLUMNS:
                row.append(np.random.rand() * 100 + np.random.randint(0,50)) # Some random values
            dummy_data_rows.append(row)
        
        dummy_df = pd.DataFrame(dummy_data_rows, columns=cols_for_dummy)
        dummy_df.to_csv(file_path, index=False)
        logging.info(f"Created dummy data with {len(dummy_data_rows)} rows at {file_path}.")
    except Exception as e:
        logging.error(f"Failed to create dummy data at {file_path}: {e}")


def start_training():
    """
    Initiates and runs the training pipeline.
    """
    try:
        # Generate a unique timestamp for this pipeline run
        pipeline_run_timestamp = TIMESTAMP
        logging.info(f"--- Main: Starting New Training Pipeline Run: {pipeline_run_timestamp} ---")

        # Setup data if needed (for local runs or first-time setup)
        setup_initial_data_if_needed()

        training_pipeline = TrainingPipeline(run_timestamp=pipeline_run_timestamp)
        training_pipeline.run_pipeline()
        
        logging.info(f"--- Main: Training Pipeline Run {pipeline_run_timestamp} Finished Successfully ---")

    except CustomException as e:
        logging.error(f"--- Main: Training Pipeline Run FAILED (CustomException) --- Error: {e}")
        # Depending on desired behavior, you might want to exit or raise further
        # print(f"Pipeline Error: {e}", file=sys.stderr)
    except FileNotFoundError as e: # Specifically catch if data setup fails critically
        logging.error(f"--- Main: Training Pipeline Run FAILED (FileNotFoundError) --- Error: {e}")
        # print(f"Pipeline Error: Missing critical file - {e}", file=sys.stderr)
    except Exception as e:
        logging.error(f"--- Main: Training Pipeline Run FAILED (Unexpected Error) --- Error: {e}")
        logging.error(traceback.format_exc())
        # print(f"Unexpected Pipeline Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    # This script is intended to run the training pipeline.
    # The Flask app (app.py) is separate for serving the UI and batch predictions.
    start_training()


import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split # Not strictly needed for unsupervised initial ingestion

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.config_entity import DataIngestionConfig
from custsegments.entity.artifact_entity import DataIngestionArtifact
from custsegments.constant.training_pipeline import (
    RAW_DATA_DIR, DATA_FILE_NAME, INGESTED_DATA_DIR, INGESTED_DATA_FILE, PROJECT_ROOT_DIR, CUSTOMER_DATA_PARENT_DIR
)
from custsegments.utils.main_utils.utils import create_directories, read_csv


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config
        # Ensure necessary directories exist for output
        create_directories([self.config.root_dir, Path(os.path.dirname(self.config.local_data_file))])

    def download_data(self) -> Path:
        """
        Downloads data from the source. In this project, it copies from a local path.
        Returns:
            Path: Path to the downloaded/copied raw data file.
        """
        try:
            # In this specific project, the data is assumed to be already present locally.
            # The 'source_url' in config might be a placeholder or for future use.
            # We'll simulate "downloading" by ensuring the file exists at the expected raw data path.
            
            # The raw data file is expected to be in customer_segments/customer_data/
            # CUSTOMER_DATA_PARENT_DIR points to customer_segments/customer_data
            # DATA_FILE_NAME is data_credit_card_customer_seg.csv
            raw_data_source_path = Path(CUSTOMER_DATA_PARENT_DIR) / DATA_FILE_NAME
            
            logging.info(f"Attempting to access raw data from: {raw_data_source_path}")

            if not raw_data_source_path.exists():
                error_msg = f"Raw data file not found at {raw_data_source_path}. Please ensure it's in the '{RAW_DATA_DIR}' directory."
                logging.error(error_msg)
                # For a real download, you might raise an exception or attempt download here.
                # For this project, we will rely on the user to place it.
                # However, the problem description implies the agent should handle moving it.
                # Let's try to find it in the original location and copy it.
                original_repo_data_path = Path(PROJECT_ROOT_DIR) / "customer_data" / "credit_card_data.csv" # Original path
                if original_repo_data_path.exists():
                    logging.info(f"Found data at original location: {original_repo_data_path}. Copying to {raw_data_source_path}")
                    os.makedirs(raw_data_source_path.parent, exist_ok=True)
                    import shutil
                    shutil.copy(original_repo_data_path, raw_data_source_path)
                    logging.info(f"Successfully copied data to {raw_data_source_path}")
                else:
                    logging.error(f"Could not find data at original location: {original_repo_data_path} either.")
                    raise FileNotFoundError(error_msg)

            # The config.local_data_file is where the *ingested* data will be stored,
            # not the raw source if they are different.
            # For this component, the primary output is the path to the *raw* data that was "ingested".
            # So, we return the path to the file in customer_data/
            
            # The task is to rebuild from scratch. The data ingestion component's role is to make the
            # raw dataset available for the subsequent pipeline steps, typically by copying it to an artifact location.
            # Let's copy the raw_data_source_path to self.config.local_data_file (which is the artifact path for ingestion)
            
            ingested_file_path = self.config.local_data_file # This is like customer_segments/artifacts/.../full_data.csv
            os.makedirs(ingested_file_path.parent, exist_ok=True)
            import shutil
            shutil.copy(raw_data_source_path, ingested_file_path)
            logging.info(f"Copied raw data from {raw_data_source_path} to ingested data path: {ingested_file_path}")

            return ingested_file_path # Return the path of the data in the artifact store

        except Exception as e:
            logging.error(f"Error during data download/access: {e}")
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main method to initiate data ingestion process.
        For unsupervised learning, we typically use the whole dataset.
        No train-test split is performed here but data is prepared for pipeline use.

        Returns:
            DataIngestionArtifact: Artifact containing paths to ingested data.
        """
        logging.info("Starting data ingestion process.")
        try:
            ingested_data_file_path = self.download_data()
            
            # For this project, we're not splitting into train/test for unsupervised learning at this stage.
            # The 'ingested_data_file_path' will serve as the main dataset for subsequent steps.

            data_ingestion_artifact = DataIngestionArtifact(
                ingested_data_path=ingested_data_file_path,
                is_ingested=True,
                message="Data ingestion completed successfully. Full dataset prepared."
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except FileNotFoundError as e:
            logging.error(f"Data ingestion failed: {e}")
            data_ingestion_artifact = DataIngestionArtifact(
                ingested_data_path=None,
                is_ingested=False,
                message=str(e)
            )
            # Depending on pipeline design, might want to raise CustomException here too
            # For now, returning an artifact indicating failure.
            raise CustomException(e, sys) from e # Re-raise to halt pipeline if data is critical
        except Exception as e:
            logging.error(f"An unexpected error occurred during data ingestion: {e}")
            raise CustomException(e, sys) from e


if __name__ == '__main__':
    # Example usage for testing the component
    # Define paths relative to the new project structure for constants
    # Assuming this script is run from customer_segments/
    
    # --- Configuration for DataIngestion ---
    # PIPELINE_ROOT is ARTIFACTS_DIR / PIPELINE_NAME
    # ARTIFACTS_DIR is "artifacts"
    # PIPELINE_NAME is "customer_segments_pipeline"
    # DATA_INGESTION_ROOT_DIR is PIPELINE_ROOT / DATA_INGESTION_DIR_NAME
    # DATA_INGESTION_DIR_NAME is "data_ingestion"
    # INGESTED_DATA_DIR is DATA_INGESTION_ROOT_DIR / "ingested_data"
    # INGESTED_DATA_FILE is "full_data.csv"
    
    # PROJECT_ROOT_DIR should be customer_segments/
    # CUSTOMER_DATA_PARENT_DIR is customer_segments/customer_data/
    # DATA_FILE_NAME is "data_credit_card_customer_seg.csv"

    # Constructing paths based on constants for clarity
    from custsegments.constant.training_pipeline import (
        ARTIFACTS_DIR, PIPELINE_NAME, DATA_INGESTION_DIR_NAME, INGESTED_DATA_FILE
    )

    # Define where the test run's artifacts will go
    test_artifacts_root = Path(PROJECT_ROOT_DIR) / ARTIFACTS_DIR / f"{PIPELINE_NAME}_test"
    
    data_ingestion_config = DataIngestionConfig(
        root_dir=test_artifacts_root / DATA_INGESTION_DIR_NAME,
        source_url="placeholder_url", # Not used as data is local
        local_data_file=test_artifacts_root / DATA_INGESTION_DIR_NAME / INGESTED_DATA_FILE, # This is the artifact path
        unzip_dir=None # Not used
    )

    # Ensure the source data file exists for the test
    # This part simulates the user having the data in the expected `customer_data` folder
    source_csv_for_test = Path(CUSTOMER_DATA_PARENT_DIR) / DATA_FILE_NAME
    
    # If the original data isn't in `customer_data`, the component should copy it from `1) Dataset/`
    # For testing, we can manually ensure it's in one of these places or create a dummy.
    # Let's assume it tries to copy from original repo structure if not in `customer_data`
    
    # Create a dummy source CSV if it doesn't exist in the original repo path for local testing
    original_dataset_path_for_test = Path(PROJECT_ROOT_DIR) / "1) Dataset" / DATA_FILE_NAME
    if not original_dataset_path_for_test.exists():
        # Create a minimal dummy CSV for testing purposes
        os.makedirs(original_dataset_path_for_test.parent, exist_ok=True)
        dummy_df_content = "CUST_ID,BALANCE,TENURE\nC001,100.0,12\nC002,200.0,10"
        with open(original_dataset_path_for_test, "w") as f:
            f.write(dummy_df_content)
        logging.info(f"Created dummy data at {original_dataset_path_for_test} for testing.")
    
    # Also ensure customer_data directory exists, though the component should handle copying if it's empty
    os.makedirs(CUSTOMER_DATA_PARENT_DIR, exist_ok=True)


    logging.info(f"Data Ingestion Config for test: {data_ingestion_config}")

    try:
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion successful. Artifact: {data_ingestion_artifact}")
        
        # Verify the output file exists
        assert data_ingestion_artifact.is_ingested
        assert data_ingestion_artifact.ingested_data_path.exists()
        logging.info(f"Ingested data file created at: {data_ingestion_artifact.ingested_data_path}")

    except CustomException as e:
        logging.error(f"Data Ingestion component test failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in Data Ingestion component test: {e}")
        import traceback
        logging.error(traceback.format_exc())

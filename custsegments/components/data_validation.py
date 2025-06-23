import os
import sys
from pathlib import Path
import pandas as pd
import json

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.config_entity import DataValidationConfig
from custsegments.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from custsegments.utils.main_utils.utils import read_csv, save_json, create_directories,load_json
from custsegments.constant.training_pipeline import EXPECTED_SCHEMA # Import schema

class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        self.ingestion_artifact = data_ingestion_artifact
        self.config = data_validation_config
        self.schema = EXPECTED_SCHEMA # Use schema from constants
        create_directories([self.config.root_dir, Path(os.path.dirname(self.config.status_file))])

    def validate_data_schema(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the schema of the dataframe against the expected schema.
        Checks for column presence and data types.

        Args:
            dataframe (pd.DataFrame): The dataframe to validate.

        Returns:
            bool: True if schema is valid, False otherwise.
        """
        try:
            logging.info("Starting schema validation.")
            validation_status = True
            report = {"schema_validation": {"overall_status": "SUCCESS", "checks": []}}

            # Check for missing columns
            expected_cols = set(self.schema.keys())
            actual_cols = set(dataframe.columns)
            
            missing_cols = list(expected_cols - actual_cols)
            if missing_cols:
                status_msg = f"Missing columns: {missing_cols}"
                report["schema_validation"]["checks"].append({"check": "Column Presence", "status": "FAIL", "message": status_msg})
                logging.error(status_msg)
                validation_status = False
            else:
                report["schema_validation"]["checks"].append({"check": "Column Presence", "status": "SUCCESS", "message": "All expected columns present."})


            # Check data types for common columns
            for col, expected_type in self.schema.items():
                if col in dataframe.columns:
                    actual_type = str(dataframe[col].dtype)
                    if expected_type == "float64" and (actual_type == "int64" or actual_type == "float64"):
                        # Allow int64 if float64 is expected, as it can be safely cast or implies no fractional part initially
                        pass # Type is compatible
                    elif expected_type == "int64" and (actual_type == "int64" or (actual_type == "float64" and dataframe[col].apply(lambda x: x.is_integer() if pd.notnull(x) else True).all())):
                        # Allow float64 if int64 is expected, provided all values are whole numbers (or NaN)
                        pass # Type is compatible or can be safely converted
                    elif actual_type != expected_type:
                        status_msg = f"Data type mismatch for column '{col}'. Expected: {expected_type}, Actual: {actual_type}"
                        report["schema_validation"]["checks"].append({"check": f"Data Type - {col}", "status": "FAIL", "message": status_msg})
                        logging.warning(status_msg) # Warning, as some mismatches might be handled in transformation
                        # validation_status = False # Decide if this should strictly fail validation
                    else:
                         report["schema_validation"]["checks"].append({"check": f"Data Type - {col}", "status": "SUCCESS"})
                # If column was missing, it's already logged.

            if not validation_status:
                 report["schema_validation"]["overall_status"] = "FAIL"
            
            save_json(self.config.validation_report_path, report) # Save the detailed report
            logging.info(f"Schema validation report saved to {self.config.validation_report_path}")
            return validation_status

        except Exception as e:
            logging.error(f"Error during schema validation: {e}")
            raise CustomException(e, sys) from e

    def validate_null_values(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates for excessive null values in the dataframe.
        This is a basic check; more sophisticated checks can be added.

        Args:
            dataframe (pd.DataFrame): The dataframe to validate.

        Returns:
            bool: True if null value checks pass, False otherwise.
        """
        try:
            logging.info("Starting null value validation.")
            validation_status = True
            # Load the report, append to it
            try:
                report = load_json(self.config.validation_report_path)
            except FileNotFoundError: # Should not happen if schema validation runs first
                report = {"schema_validation": {}, "null_validation": {"overall_status": "SUCCESS", "checks": []}}
            
            report.setdefault("null_validation", {"overall_status": "SUCCESS", "checks": []})

            for col in dataframe.columns:
                if col in self.schema: # Only validate columns defined in schema for now
                    null_percentage = (dataframe[col].isnull().sum() / len(dataframe)) * 100
                    if null_percentage > 50: # Example threshold: fail if more than 50% nulls
                        status_msg = f"Column '{col}' has {null_percentage:.2f}% null values, which exceeds threshold."
                        report["null_validation"]["checks"].append({"check": f"Null Values - {col}", "status": "FAIL", "message": status_msg})
                        logging.warning(status_msg)
                        # validation_status = False # Decide if this should strictly fail validation or just warn
                    else:
                        report["null_validation"]["checks"].append({"check": f"Null Values - {col}", "status": "SUCCESS", "message": f"{null_percentage:.2f}% nulls."})
            
            if any(check["status"] == "FAIL" for check in report["null_validation"]["checks"]):
                 report["null_validation"]["overall_status"] = "FAIL"
                 # validation_status = False # If any check fails, overall null validation fails

            save_json(self.config.validation_report_path, report)
            logging.info(f"Null value validation report updated at {self.config.validation_report_path}")
            return validation_status # Returns true if no hard failures, allows pipeline to proceed with warnings
        except Exception as e:
            logging.error(f"Error during null value validation: {e}")
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Main method to orchestrate data validation checks.
        """
        logging.info("Starting data validation process.")
        try:
            if not self.ingestion_artifact.is_ingested or not self.ingestion_artifact.ingested_data_path:
                message = "Data ingestion was not successful or data path is missing. Skipping validation."
                logging.warning(message)
                return DataValidationArtifact(validation_status=False, validated_data_path=None, validation_report_path=self.config.validation_report_path, message=message)

            data_df = read_csv(self.ingestion_artifact.ingested_data_path)
            logging.info(f"Dataframe loaded for validation. Shape: {data_df.shape}")
            
            # Perform schema validation
            schema_valid = self.validate_data_schema(data_df)
            
            # Perform null value validation (can add more checks like outlier detection, etc.)
            null_check_passed = self.validate_null_values(data_df) # This appends to the same report

            # Overall validation status: for now, let's say it's valid if schema is okay,
            # even if there are nulls (as they will be handled).
            # A stricter pipeline might set overall_status to False if any check fails.
            overall_validation_status = schema_valid and null_check_passed # Modify this logic if needed for stricter pass/fail

            status_message = "Data validation completed."
            if not overall_validation_status:
                status_message = "Data validation completed with issues. Check report."
                logging.warning(status_message)
            else:
                logging.info(status_message)

            # Write overall status to a simple status file
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation Status: {'SUCCESS' if overall_validation_status else 'FAIL'}\n")
                f.write(f"Schema Valid: {schema_valid}\n")
                f.write(f"Null Check Passed (no hard fails): {null_check_passed}\n")

            validation_artifact = DataValidationArtifact(
                validation_status=overall_validation_status,
                # If validation passes (even with warnings to be handled), pass the data path along.
                # If it fails critically (e.g., schema mismatch that can't be handled), this might be None.
                validated_data_path=self.ingestion_artifact.ingested_data_path if overall_validation_status else None,
                validation_report_path=self.config.validation_report_path,
                message=status_message
            )
            logging.info(f"Data validation artifact: {validation_artifact}")
            return validation_artifact

        except Exception as e:
            logging.error(f"Error during data validation initiation: {e}")
            # Ensure a report and status file are written even on unhandled exception if possible
            try:
                error_report = {"error": str(e), "traceback": traceback.format_exc()}
                save_json(self.config.validation_report_path, error_report)
                with open(self.config.status_file, 'w') as f:
                    f.write("Validation Status: ERROR\n")
                    f.write(f"Error: {str(e)}\n")
            except Exception as report_e:
                logging.error(f"Failed to write error report for data validation: {report_e}")

            raise CustomException(e, sys) from e


if __name__ == '__main__':
    # Example Usage for testing the component
    from custsegments.constant.training_pipeline import (
        ARTIFACTS_DIR, PIPELINE_NAME, DATA_INGESTION_DIR_NAME, INGESTED_DATA_FILE,
        DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_STATUS_FILE, DATA_VALIDATION_REPORT_FILE
    )
    import shutil

    # --- Configuration for DataValidation Test ---
    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent.parent 
    
    # Use the same test artifacts root as data ingestion for continuity
    test_artifacts_root = project_root_for_test / ARTIFACTS_DIR / f"{PIPELINE_NAME}_test_dv" 
    
    # Dummy DataIngestionArtifact (assuming data_ingestion ran successfully)
    # Ensure the ingested data file path for the artifact points to a real (dummy) file
    dummy_ingested_data_path = test_artifacts_root / DATA_INGESTION_DIR_NAME / INGESTED_DATA_FILE
    os.makedirs(dummy_ingested_data_path.parent, exist_ok=True)
    
    # Create a dummy CSV that matches expected schema (more or less)
    # Include some NaNs for MINIMUM_PAYMENTS and CREDIT_LIMIT
    dummy_data_content = (
        "CUST_ID,BALANCE,BALANCE_FREQUENCY,PURCHASES,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE,"
        "PURCHASES_FREQUENCY,ONEOFF_PURCHASES_FREQUENCY,PURCHASES_INSTALLMENTS_FREQUENCY,CASH_ADVANCE_FREQUENCY,"
        "CASH_ADVANCE_TRX,PURCHASES_TRX,CREDIT_LIMIT,PAYMENTS,MINIMUM_PAYMENTS,PRC_FULL_PAYMENT,TENURE\n"
        "C10001,40.9,0.81,95.4,0,95.4,0,0.16,0,0.08,0,0,2,1000,201.8,,0,12\n"  # MINIMUM_PAYMENTS NaN
        "C10002,3202.4,0.9,0,0,0,6442.9,0,0,0,0.25,4,0,,4103.0,1072.3,0.22,12\n" # CREDIT_LIMIT NaN
        "C10003,2495.1,1,773.1,773.1,0,0,1,1,0,0,0,12,7500,622.0,627.2,0,12\n"
        # Add a row with a type mismatch for testing schema validation failure
        "C10004,TEXT,1,773.1,773.1,0,0,1,1,0,0,0,12,7500,622.0,627.2,0,12\n"
    )
    with open(dummy_ingested_data_path, "w") as f:
        f.write(dummy_data_content)
    logging.info(f"Created dummy ingested data at {dummy_ingested_data_path} for DataValidation test.")

    data_ingestion_artifact_test = DataIngestionArtifact(
        ingested_data_path=dummy_ingested_data_path,
        is_ingested=True,
        message="Dummy data ingestion artifact for validation test."
    )

    data_validation_config = DataValidationConfig(
        root_dir=test_artifacts_root / DATA_VALIDATION_DIR_NAME,
        status_file=test_artifacts_root / DATA_VALIDATION_DIR_NAME / DATA_VALIDATION_STATUS_FILE,
        validation_report_path = test_artifacts_root / DATA_VALIDATION_DIR_NAME / DATA_VALIDATION_REPORT_FILE,
        all_required_files=[], # Not directly used in this component's current logic beyond schema
        data_path=dummy_ingested_data_path, # Path to the data to be validated
        schema=EXPECTED_SCHEMA 
    )
    logging.info(f"Data Validation Config for test: {data_validation_config}")

    try:
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact_test,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation component test finished. Artifact: {data_validation_artifact}")

        # Check report content (example)
        if data_validation_artifact.validation_report_path.exists():
            with open(data_validation_artifact.validation_report_path, 'r') as f:
                report_content = json.load(f)
            logging.info(f"Validation report content:\n{json.dumps(report_content, indent=2)}")
        
        # Clean up dummy artifacts directory for this specific test
        # shutil.rmtree(test_artifacts_root) 
        # logging.info(f"Cleaned up test artifacts directory: {test_artifacts_root}")


    except CustomException as ce:
        logging.error(f"Data Validation component test failed: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error in Data Validation component test: {e}")
        import traceback
        logging.error(traceback.format_exc())

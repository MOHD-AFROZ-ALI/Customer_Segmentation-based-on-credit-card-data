import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.config_entity import DataTransformationConfig
from custsegments.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from custsegments.utils.main_utils.utils import save_object, read_csv, save_csv, create_directories,save_json
from custsegments.constant.training_pipeline import COLUMNS_TO_DROP, NUMERICAL_COLUMNS 

class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        self.validation_artifact = data_validation_artifact
        self.config = data_transformation_config
        create_directories([self.config.root_dir, Path(os.path.dirname(self.config.preprocessor_obj_file_path)), Path(os.path.dirname(self.config.transformed_data_path))])

    def get_preprocessor_object(self) -> Pipeline:
        """
        Creates and returns a scikit-learn Pipeline for preprocessing numerical features.
        This pipeline includes:
        1. Imputation of missing values using the median.
        2. Scaling features using StandardScaler.
        """
        try:
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
                ('scaler', StandardScaler())
            ])
            logging.info("Numerical preprocessing pipeline created with Median Imputer and StandardScaler.")
            return numerical_pipeline

        except Exception as e:
            logging.error(f"Error creating preprocessor object: {e}")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Orchestrates the data transformation process.
        Reads validated data, applies preprocessing, and saves the transformed data
        and the preprocessor object.
        """
        try:
            if not self.validation_artifact.validation_status or not self.validation_artifact.validated_data_path:
                message = "Data validation was not successful or validated data path is missing. Skipping transformation."
                logging.warning(message)
                return DataTransformationArtifact(
                    transformed_data_path=None,
                    preprocessor_object_path=None,
                    is_transformed=False,
                    message=message
                )

            logging.info("Starting data transformation process.")
            
            # Load the validated data
            df = read_csv(self.validation_artifact.validated_data_path)
            logging.info(f"Loaded validated data from: {self.validation_artifact.validated_data_path}")

            # Drop the identifier column if it exists and is specified in COLUMNS_TO_DROP
            df_transformed = df.copy()
            for col_to_drop in COLUMNS_TO_DROP:
                if col_to_drop in df_transformed.columns:
                    df_transformed.drop(columns=[col_to_drop], inplace=True)
                    logging.info(f"Dropped column: {col_to_drop}")
            
            # Identify numerical columns for transformation (all features except CUST_ID are numerical in this dataset)
            # NUMERICAL_COLUMNS should be defined in constants and verified
            numerical_features = [col for col in NUMERICAL_COLUMNS if col in df_transformed.columns]
            if not numerical_features:
                raise ValueError("No numerical features found for transformation. Check NUMERICAL_COLUMNS constant.")

            logging.info(f"Numerical features to be transformed: {numerical_features}")

            # Get the preprocessor pipeline
            preprocessor = self.get_preprocessor_object()

            # Apply the preprocessing pipeline to numerical features
            # Note: fit_transform expects a DataFrame or 2D array.
            df_transformed[numerical_features] = preprocessor.fit_transform(df_transformed[numerical_features])
            # imputer = SimpleImputer(strategy='median')
            # imputed_array = imputer.fit_transform(df_transformed[numerical_features])
            # imputed_df = pd.DataFrame(imputed_array, columns=numerical_features, index=df_transformed.index)
            # df_imputed = df_transformed.copy()
            # df_imputed[numerical_features] = imputed_df

            # # Save the imputed (but unscaled) data as an intermediate artifact
            # imputed_data_path = str(self.config.transformed_data_path).replace(".csv", "_imputed.csv")
            # save_csv(df=df_imputed, file_path=imputed_data_path)
            # logging.info(f"Imputed (unscaled) data saved to: {imputed_data_path}")

            # # --- Now apply scaling ---
            # scaler = StandardScaler()
            # scaled_array = scaler.fit_transform(imputed_df)
            # df_transformed[numerical_features] = scaled_array
            
            # logging.info("Data transformed using the preprocessing pipeline.")
            
            # Save the preprocessor object
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            logging.info(f"Preprocessor object saved to: {self.config.preprocessor_obj_file_path}")

            # Save the transformed dataframe
            # The path for transformed data should be config.transformed_data_path
            save_csv(df=df_transformed, file_path=self.config.transformed_data_path)
            logging.info(f"Transformed data saved to: {self.config.transformed_data_path}")

            transformation_artifact = DataTransformationArtifact(
                transformed_data_path=self.config.transformed_data_path,
                preprocessor_object_path=self.config.preprocessor_obj_file_path,
                #imputed_data_path=imputed_data_path,  # Save the imputed data path
                is_transformed=True,
                message="Data transformation completed successfully."
            )
            logging.info(f"Data transformation artifact: {transformation_artifact}")
            return transformation_artifact

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys) from e


if __name__ == '__main__':
    # Example Usage for testing the component
    from custsegments.constant.training_pipeline import (
        ARTIFACTS_DIR, PIPELINE_NAME, DATA_INGESTION_DIR_NAME, INGESTED_DATA_FILE,
        DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_STATUS_FILE, DATA_VALIDATION_REPORT_FILE,
        DATA_TRANSFORMATION_DIR_NAME, PREPROCESSOR_OBJECT_FILE_NAME, TRANSFORMED_DATA_FILE
    )
    import shutil

    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent.parent

    test_artifacts_root = project_root_for_test / ARTIFACTS_DIR / f"{PIPELINE_NAME}_test_dt"

    # --- Dummy DataValidationArtifact (assuming data_validation ran successfully) ---
    # This path should point to a (dummy) validated data file
    dummy_validated_data_path = test_artifacts_root / DATA_VALIDATION_DIR_NAME / "validated_data.csv" # Example name
    os.makedirs(dummy_validated_data_path.parent, exist_ok=True)

    # Create a dummy CSV that mimics what validated data might look like
    # (e.g., after basic cleaning or just the raw ingested if validation only checked schema/nulls)
    # For this test, let's assume it's similar to the ingested data but "validated"
    # Using a subset of columns from EXPECTED_SCHEMA for simplicity in dummy data
    dummy_data_content_validated = (
        "CUST_ID,BALANCE,PURCHASES,CREDIT_LIMIT,MINIMUM_PAYMENTS,TENURE\n"
        "C10001,40.9,95.4,1000,,12\n"  # MINIMUM_PAYMENTS NaN
        "C10002,3202.4,0,,4103.0,12\n" # CREDIT_LIMIT NaN
        "C10003,2495.1,773.1,7500,622.0,12\n"
        "C10005,817.7,16.0,1200,244.7,12\n" # A more complete row
    )
    with open(dummy_validated_data_path, "w") as f:
        f.write(dummy_data_content_validated)
    logging.info(f"Created dummy validated data at {dummy_validated_data_path} for DataTransformation test.")

    data_validation_artifact_test = DataValidationArtifact(
        validation_status=True,
        validated_data_path=dummy_validated_data_path,
        validation_report_path=test_artifacts_root / DATA_VALIDATION_DIR_NAME / DATA_VALIDATION_REPORT_FILE, # Dummy path
        message="Dummy data validation artifact for transformation test."
    )
    # Ensure report path dir exists for the dummy artifact
    os.makedirs(data_validation_artifact_test.validation_report_path.parent, exist_ok=True)
    save_json(data_validation_artifact_test.validation_report_path, {"status": "dummy report"})


    # --- Configuration for DataTransformation ---
    data_transformation_config = DataTransformationConfig(
        root_dir=test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME,
        data_path=data_validation_artifact_test.validated_data_path, # Input for transformation
        preprocessor_obj_file_path=test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME / PREPROCESSOR_OBJECT_FILE_NAME,
        transformed_data_path=test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME / TRANSFORMED_DATA_FILE
    )
    logging.info(f"Data Transformation Config for test: {data_transformation_config}")

    try:
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact_test,
            data_transformation_config=data_transformation_config
        )
        transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation component test finished. Artifact: {transformation_artifact}")

        # Verify outputs
        assert transformation_artifact.is_transformed
        assert transformation_artifact.preprocessor_object_path.exists()
        assert transformation_artifact.transformed_data_path.exists()
        
        logging.info(f"Preprocessor saved at: {transformation_artifact.preprocessor_object_path}")
        logging.info(f"Transformed data saved at: {transformation_artifact.transformed_data_path}")

        # Optionally, load and inspect the transformed data
        transformed_df = pd.read_csv(transformation_artifact.transformed_data_path)
        logging.info(f"Transformed DataFrame head:\n{transformed_df.head()}")
        assert not transformed_df.isnull().values.any(), "Transformed data should not have NaNs after imputation."
        # Check if CUST_ID is dropped (it should be)
        assert "CUST_ID" not in transformed_df.columns, "CUST_ID should have been dropped."


        # Clean up (optional)
        # if test_artifacts_root.exists():
        #    shutil.rmtree(test_artifacts_root)
        #    logging.info(f"Cleaned up test artifacts directory: {test_artifacts_root}")

    except CustomException as ce:
        logging.error(f"Data Transformation component test failed: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error in Data Transformation component test: {e}")
        import traceback
        logging.error(traceback.format_exc())

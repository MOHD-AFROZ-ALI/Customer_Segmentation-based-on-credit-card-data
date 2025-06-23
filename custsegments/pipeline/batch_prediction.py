import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import traceback

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.config_entity import BatchPredictionConfig
from custsegments.entity.artifact_entity import BatchPredictionArtifact
from custsegments.utils.main_utils.utils import load_object, read_csv, save_csv, create_directories,save_object
from custsegments.constant.training_pipeline import COLUMNS_TO_DROP # For dropping CUST_ID if present in uploaded file

class BatchPredictionPipeline:
    def __init__(self, config: BatchPredictionConfig):
        self.config = config
        # Ensure output directory exists
        create_directories([Path(os.path.dirname(self.config.output_csv_path))])

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch prediction on the input DataFrame.
        Loads the preprocessor and model, transforms the data, and predicts clusters.
        Returns the DataFrame with an added 'CLUSTER' column.
        """
        try:
            logging.info("Starting batch prediction...")

            # Load preprocessor
            preprocessor_path = Path(self.config.preprocessor_path)
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info(f"Preprocessor loaded from {preprocessor_path}")

            # Load model
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model = load_object(file_path=model_path)
            logging.info(f"Model loaded from {model_path}")

            # Prepare input data
            df_to_predict = input_df.copy()
            
            # Store CUST_ID if it exists, then drop it for prediction
            cust_ids = None
            if 'CUST_ID' in df_to_predict.columns: # Assuming CUST_ID is the identifier
                cust_ids = df_to_predict['CUST_ID'].copy()
                # Drop specified columns (like CUST_ID) before transformation
                cols_to_drop_actual = [col for col in COLUMNS_TO_DROP if col in df_to_predict.columns]
                if cols_to_drop_actual:
                    df_to_predict.drop(columns=cols_to_drop_actual, inplace=True)
                    logging.info(f"Dropped columns for prediction: {cols_to_drop_actual}")
            
            # Ensure columns match the order/features the preprocessor was trained on
            # This assumes the preprocessor handles selection of appropriate columns or
            # that the input_df already matches the schema expected by the preprocessor.
            # For robustness, explicit column selection based on preprocessor's known features is better.
            # If preprocessor is a Pipeline of (Imputer, Scaler) on all numerical features:
            
            numerical_features_for_transform = [col for col in df_to_predict.columns if col in preprocessor.feature_names_in_] if hasattr(preprocessor, 'feature_names_in_') else df_to_predict.columns.tolist()
            
            if not all(col in df_to_predict.columns for col in numerical_features_for_transform):
                 missing_cols_for_pred = set(numerical_features_for_transform) - set(df_to_predict.columns)
                 raise ValueError(f"Input data for prediction is missing required columns: {missing_cols_for_pred}")

            # Transform data
            # Note: preprocessor.transform expects data in the same format/columns as during fit.
            # If preprocessor was fit on a DataFrame with specific columns, ensure df_to_predict has them.
            # A ColumnTransformer based preprocessor would be more robust here.
            # Assuming the SimpleImputer + StandardScaler pipeline applies to all provided columns.
            
            # Handle cases where preprocessor might be a simple scikit-learn object (e.g. StandardScaler)
            # or a Pipeline.
            if hasattr(preprocessor, 'transform') and callable(preprocessor.transform):
                transformed_data_array = preprocessor.transform(df_to_predict[numerical_features_for_transform])
            else:
                raise TypeError("Preprocessor does not have a callable 'transform' method.")

            transformed_df = pd.DataFrame(transformed_data_array, columns=numerical_features_for_transform, index=df_to_predict.index)
            logging.info("Input data transformed for prediction.")

            # Make predictions
            if hasattr(model, 'predict') and callable(model.predict):
                cluster_labels = model.predict(transformed_df)
            elif hasattr(model, 'labels_') and isinstance(model, DBSCAN): # DBSCAN assigns labels during fit, for batch predict need to re-fit or use fit_predict
                # For true batch prediction with DBSCAN, you'd typically use fit_predict
                # or a model that can predict on new data without re-fitting (like KMeans).
                # If the requirement is to assign new data to existing DBSCAN clusters,
                # that's more complex (e.g. finding nearest core point).
                # For simplicity here, if it's DBSCAN, we'll log a warning or re-fit_predict.
                logging.warning("DBSCAN loaded. For batch prediction, it will re-cluster the new data using fit_predict.")
                cluster_labels = model.fit_predict(transformed_df)
            else:
                raise TypeError("Model does not have a callable 'predict' or 'fit_predict' method.")
            
            logging.info("Predictions made.")

            # Add cluster labels to the original DataFrame (or a copy of it)
            # If CUST_ID was present, we use the original input_df to preserve it.
            output_df = input_df.copy()
            output_df['CLUSTER'] = cluster_labels
            
            logging.info("Cluster labels added to the output DataFrame.")
            return output_df

        except FileNotFoundError as e:
            logging.error(f"Batch prediction failed due to missing file: {e}")
            raise CustomException(e, sys) from e
        except ValueError as e: # Catch missing columns error
            logging.error(f"Batch prediction failed due to data error: {e}")
            raise CustomException(e, sys) from e
        except Exception as e:
            logging.error(f"Error during batch prediction: {e}\n{traceback.format_exc()}")
            raise CustomException(e, sys) from e

    def initiate_batch_prediction(self) -> BatchPredictionArtifact:
        """
        Reads input CSV, performs predictions, and saves the results.
        """
        logging.info("Starting batch prediction pipeline...")
        try:
            input_df = read_csv(self.config.input_csv_path)
            logging.info(f"Loaded input data from {self.config.input_csv_path} for batch prediction. Shape: {input_df.shape}")

            predicted_df = self.predict(input_df)

            save_csv(df=predicted_df, file_path=self.config.output_csv_path)
            logging.info(f"Batch prediction results saved to {self.config.output_csv_path}")

            artifact = BatchPredictionArtifact(
                output_file_path=self.config.output_csv_path,
                num_records_processed=len(predicted_df),
                is_prediction_successful=True,
                message="Batch prediction completed successfully."
            )
            logging.info(f"Batch prediction artifact: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"Batch prediction pipeline failed: {e}")
            # Create a failure artifact
            artifact = BatchPredictionArtifact(
                output_file_path=None,
                num_records_processed=0,
                is_prediction_successful=False,
                message=f"Batch prediction failed: {str(e)}"
            )
            # Depending on desired behavior, might re-raise or just return failure artifact
            raise CustomException(e, sys) from e


if __name__ == '__main__':
    # Example usage for testing BatchPredictionPipeline
    from custsegments.constant.training_pipeline import (
        PROJECT_ROOT_DIR, FINAL_MODEL_DIR, PREPROCESSOR_OBJECT_FILE_NAME, MODEL_OBJECT_FILE_NAME,
        PREDICTION_OUTPUT_DIR, BATCH_PREDICTION_OUTPUT_FILE, NUMERICAL_COLUMNS
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans
    import shutil

    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent.parent

    # --- Setup dummy model and preprocessor (as if from a training run) ---
    # These should be placed where BatchPredictionConfig expects them
    # FINAL_MODEL_DIR = "final_model" (relative to project_root)
    # PREDICTION_OUTPUT_DIR = "prediction_output" (relative to project_root)

    dummy_final_model_dir = project_root_for_test / FINAL_MODEL_DIR
    os.makedirs(dummy_final_model_dir, exist_ok=True)
    
    dummy_model_path = dummy_final_model_dir / MODEL_OBJECT_FILE_NAME
    dummy_preprocessor_path = dummy_final_model_dir / PREPROCESSOR_OBJECT_FILE_NAME

    # Create and save a dummy preprocessor
    dummy_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Fit preprocessor on data that has the same columns as NUMERICAL_COLUMNS
    # Ensure NUMERICAL_COLUMNS is not empty for dummy data generation
    if not NUMERICAL_COLUMNS:
        raise ValueError("NUMERICAL_COLUMNS constant is empty. Cannot create dummy data for preprocessor fitting.")
        
    dummy_preprocessor_fit_data = pd.DataFrame(np.random.rand(10, len(NUMERICAL_COLUMNS)), columns=NUMERICAL_COLUMNS)
    dummy_preprocessor.fit(dummy_preprocessor_fit_data)
    save_object(dummy_preprocessor_path, dummy_preprocessor)
    logging.info(f"Saved dummy preprocessor to {dummy_preprocessor_path}")

    # Create and save a dummy KMeans model
    dummy_kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    # Fit with data that has the same features as preprocessor output
    dummy_kmeans_fit_data = pd.DataFrame(np.random.randn(10, len(NUMERICAL_COLUMNS)), columns=NUMERICAL_COLUMNS)
    dummy_kmeans.fit(dummy_kmeans_fit_data)
    save_object(dummy_model_path, dummy_kmeans)
    logging.info(f"Saved dummy KMeans model to {dummy_model_path}")


    # --- Create dummy input CSV for batch prediction ---
    dummy_input_dir = project_root_for_test / "temp_batch_input"
    os.makedirs(dummy_input_dir, exist_ok=True)
    dummy_input_csv_path = dummy_input_dir / "batch_input_data.csv"
    
    input_data_cols = ["CUST_ID"] + NUMERICAL_COLUMNS # Include CUST_ID
    dummy_input_rows = [
        ["NEW_CUST1"] + [np.random.rand()*2000 + i for i, _ in enumerate(NUMERICAL_COLUMNS)],
        ["NEW_CUST2"] + [np.random.rand()*3000 + i for i, _ in enumerate(NUMERICAL_COLUMNS)],
        ["NEW_CUST3"] + [np.random.rand()*1000 + i for i, _ in enumerate(NUMERICAL_COLUMNS)]
    ]
    dummy_input_df = pd.DataFrame(dummy_input_rows, columns=input_data_cols)
    dummy_input_df.to_csv(dummy_input_csv_path, index=False)
    logging.info(f"Created dummy input CSV for batch prediction at {dummy_input_csv_path}")


    # --- BatchPredictionConfig ---
    dummy_output_dir = project_root_for_test / PREDICTION_OUTPUT_DIR
    os.makedirs(dummy_output_dir, exist_ok=True)
    
    batch_pred_config = BatchPredictionConfig(
        input_csv_path=dummy_input_csv_path,
        output_csv_path=dummy_output_dir / BATCH_PREDICTION_OUTPUT_FILE,
        model_path=dummy_model_path,
        preprocessor_path=dummy_preprocessor_path
    )
    logging.info(f"Batch Prediction Config for test: {batch_pred_config}")

    try:
        batch_pipeline = BatchPredictionPipeline(config=batch_pred_config)
        batch_artifact = batch_pipeline.initiate_batch_prediction()
        logging.info(f"Batch Prediction Pipeline test finished. Artifact: {batch_artifact}")

        assert batch_artifact.is_prediction_successful
        assert batch_artifact.output_file_path.exists()
        logging.info(f"Batch predictions saved to: {batch_artifact.output_file_path}")

        # Verify output content
        output_df_check = pd.read_csv(batch_artifact.output_file_path)
        assert 'CLUSTER' in output_df_check.columns
        assert len(output_df_check) == len(dummy_input_df)
        logging.info(f"Output CSV content head:\n{output_df_check.head()}")
        
        # Clean up dummy input and potentially output if desired
        # os.remove(dummy_input_csv_path)
        # os.rmdir(dummy_input_dir)
        # To keep output for inspection, don't delete output_csv_path or dummy_output_dir here.

    except CustomException as ce:
        logging.error(f"BatchPredictionPipeline test failed: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error in BatchPredictionPipeline test: {e}")
        logging.error(traceback.format_exc())

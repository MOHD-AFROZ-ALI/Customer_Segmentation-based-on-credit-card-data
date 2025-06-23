import sys
import os
import numpy as np
import pandas as pd
import traceback
from pathlib import Path

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException

from custsegments.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelTrainerConfig, ClusteringEvaluationConfig
)
from custsegments.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact,
    ModelTrainerArtifact, ClusteringEvaluationArtifact
)

from custsegments.components.data_ingestion import DataIngestion
from custsegments.components.data_validation import DataValidation
from custsegments.components.data_transformation import DataTransformation
from custsegments.components.model_trainer import ModelTrainer
from custsegments.components.clustering_evaluation import ClusteringEvaluation
from custsegments.components.eda_analysis import EDAAnalysis # Import EDA component

# Import constants for paths
from custsegments.constant.training_pipeline import (
    PROJECT_ROOT_DIR, ARTIFACTS_DIR, PIPELINE_NAME, TIMESTAMP,
    DATA_INGESTION_DIR_NAME, INGESTED_DATA_FILE,
    DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_STATUS_FILE, DATA_VALIDATION_REPORT_FILE,
    DATA_TRANSFORMATION_DIR_NAME, PREPROCESSOR_OBJECT_FILE_NAME, TRANSFORMED_DATA_FILE,
    MODEL_TRAINER_DIR_NAME, MODEL_OBJECT_FILE_NAME,
    CLUSTERING_EVALUATION_DIR_NAME, MODEL_EVALUATION_METRICS_FILE,
    EDA_ANALYSIS_ROOT_DIR as DEFAULT_EDA_ROOT_DIR_CONST, # For EDA component config
    EDA_PLOTS_DIR as DEFAULT_EDA_PLOTS_DIR_CONST,
    EDA_REPORT_FILE as DEFAULT_EDA_REPORT_FILE_CONST,
    # Model specific params from constants if not overridden by a config file
    KMEANS_N_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
)
from custsegments.constant.training_pipeline import EDA_ANALYSIS_DIR_NAME


class TrainingPipeline:
    def __init__(self, run_timestamp: str = TIMESTAMP):
        self.run_timestamp = run_timestamp
        self.artifacts_root = Path(PROJECT_ROOT_DIR) / ARTIFACTS_DIR / PIPELINE_NAME / self.run_timestamp
        logging.info(f"Pipeline artifacts root directory set to: {self.artifacts_root}")

        # --- Configuration Setup for each component ---
        # These would ideally be loaded from a YAML or other config management system.
        # For now, we define them directly using constants.

        # Data Ingestion Config
        self.data_ingestion_config = DataIngestionConfig(
            root_dir=self.artifacts_root / DATA_INGESTION_DIR_NAME,
            source_url="local_dataset", # Placeholder, actual logic handles local file
            local_data_file=self.artifacts_root / DATA_INGESTION_DIR_NAME / INGESTED_DATA_FILE,
            unzip_dir=None # Not used for this project
        )

        # Data Validation Config
        self.data_validation_config = DataValidationConfig(
            root_dir=self.artifacts_root / DATA_VALIDATION_DIR_NAME,
            status_file=self.artifacts_root / DATA_VALIDATION_DIR_NAME / DATA_VALIDATION_STATUS_FILE,
            validation_report_path=self.artifacts_root / DATA_VALIDATION_DIR_NAME / DATA_VALIDATION_REPORT_FILE,
            all_required_files=[], # Not strictly used by current validation logic
            data_path=self.data_ingestion_config.local_data_file, # Input for validation
            schema= {} # EXPECTED_SCHEMA is loaded within the component from constants
        )
        
        # Data Transformation Config
        self.data_transformation_config = DataTransformationConfig(
            root_dir=self.artifacts_root / DATA_TRANSFORMATION_DIR_NAME,
            data_path=self.data_ingestion_config.local_data_file, # This should be validated_data_path from DV artifact
            preprocessor_obj_file_path=self.artifacts_root / DATA_TRANSFORMATION_DIR_NAME / PREPROCESSOR_OBJECT_FILE_NAME,
            transformed_data_path=self.artifacts_root / DATA_TRANSFORMATION_DIR_NAME / TRANSFORMED_DATA_FILE
        )

        # EDA Analysis Config (paths for its outputs)
        # EDA component might run on transformed data.
        # Its outputs (plots, report) will be stored within its own artifact directory.
        self.eda_analysis_output_dir = self.artifacts_root / EDA_ANALYSIS_DIR_NAME # DEFAULT_EDA_ROOT_DIR_CONST  #.name Use .name to get 'eda_analysis'
        self.eda_plots_dir = self.eda_analysis_output_dir / "plots" # DEFAULT_EDA_PLOTS_DIR_CONST   #.name # 'plots'
        self.eda_report_file = self.eda_analysis_output_dir / "eda_report.txt" #DEFAULT_EDA_REPORT_FILE_CONST  #.name # 'eda_report.txt'
        
        # Model Trainer Config
        self.model_trainer_config = ModelTrainerConfig(
            root_dir=self.artifacts_root / MODEL_TRAINER_DIR_NAME,
            transformed_data_path=self.data_transformation_config.transformed_data_path,
            model_name=MODEL_OBJECT_FILE_NAME, # e.g., "model.pkl"
            model_save_path=self.artifacts_root / MODEL_TRAINER_DIR_NAME / MODEL_OBJECT_FILE_NAME,
            n_clusters_kmeans=KMEANS_N_CLUSTERS, # Default, can be tuned
            eps_dbscan=DBSCAN_EPS,             # Default, can be tuned
            min_samples_dbscan=DBSCAN_MIN_SAMPLES  # Default, can be tuned
        )

        # Clustering Evaluation Config
        self.clustering_evaluation_config = ClusteringEvaluationConfig(
            root_dir=self.artifacts_root / CLUSTERING_EVALUATION_DIR_NAME,
            model_path=self.model_trainer_config.model_save_path, # Path to the primary saved model
            preprocessor_path=self.data_transformation_config.preprocessor_obj_file_path,
            data_path=self.data_transformation_config.transformed_data_path, # Transformed data for metrics
            metrics_file_path=self.artifacts_root / CLUSTERING_EVALUATION_DIR_NAME / MODEL_EVALUATION_METRICS_FILE,
            # DBSCAN params for evaluation component to re-train/evaluate (ideally from a shared config or ModelTrainerArtifact)
            #eps_dbscan=self.model_trainer_config.eps_dbscan,
            #min_samples_dbscan=self.model_trainer_config.min_samples_dbscan,
            insights_file_path=self.artifacts_root / CLUSTERING_EVALUATION_DIR_NAME / "clustering_insights.txt" # For textual insights
        )


    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting Data Ingestion stage in pipeline...")
        data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
        artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion stage completed.")
        return artifact

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Starting Data Validation stage in pipeline...")
        # Update config with the actual ingested data path
        self.data_validation_config.data_path = data_ingestion_artifact.ingested_data_path
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=self.data_validation_config
        )
        artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation stage completed.")
        return artifact

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        logging.info("Starting Data Transformation stage in pipeline...")
        # Update config with the actual validated data path
        self.data_transformation_config.data_path = data_validation_artifact.validated_data_path
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=self.data_transformation_config
        )
        artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation stage completed.")
        return artifact

    def start_eda_analysis(self, data_transformation_artifact: DataTransformationArtifact):
        logging.info("Starting EDA Analysis stage in pipeline...")
        eda_analysis = EDAAnalysis(
            data_transformation_artifact=data_transformation_artifact,
            eda_output_plots_dir=self.eda_plots_dir, # Pass the specific plots dir for this run
            eda_report_file_path=self.eda_report_file   # Pass the specific report file for this run
        )
        eda_analysis.initiate_eda_analysis() # EDA outputs are files, not a structured artifact in this design
        logging.info("EDA Analysis stage completed.")
        # No artifact returned by this EDA component as per current design

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("Starting Model Training stage in pipeline...")
        # Update config with the actual transformed data path
        self.model_trainer_config.transformed_data_path = data_transformation_artifact.transformed_data_path
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=self.model_trainer_config
        )
        artifact = model_trainer.initiate_model_training()
        logging.info("Model Training stage completed.")
        return artifact

    def start_clustering_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact) -> ClusteringEvaluationArtifact:
        logging.info("Starting Clustering Evaluation stage in pipeline...")
        # Update config paths
        self.clustering_evaluation_config.model_path = model_trainer_artifact.trained_model_path
        self.clustering_evaluation_config.preprocessor_path = data_transformation_artifact.preprocessor_object_path
        self.clustering_evaluation_config.data_path = data_transformation_artifact.transformed_data_path
        
        clustering_evaluation = ClusteringEvaluation(
            model_trainer_artifact=model_trainer_artifact,
            data_transformation_artifact=data_transformation_artifact,
            clustering_evaluation_config=self.clustering_evaluation_config,
            #model_trainer_config=self.model_trainer_config # Pass for DBSCAN params -currently not needed because we are using eps_dbscan ,min_samples_dbscan
        )
        artifact = clustering_evaluation.initiate_clustering_evaluation()
        logging.info("Clustering Evaluation stage completed.")
        return artifact

    def run_pipeline(self):
        """Runs the full training pipeline."""
        logging.info(f"Training pipeline started for run: {self.run_timestamp}")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            if not data_ingestion_artifact.is_ingested:
                raise Exception("Data Ingestion failed. Halting pipeline.")

            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            if not data_validation_artifact.validation_status:
                # Depending on strictness, might halt or proceed with caution
                logging.error("Data Validation failed critical checks. Halting pipeline.")
                # raise Exception("Data Validation failed critical checks. Halting pipeline.")
                # For now, let's try to proceed if it's just warnings, but fail if critical.
                # The component itself should raise CustomException if it's a critical failure.
                # If it returns validation_status=False, it means critical failure.
                raise Exception("Data Validation failed. Halting pipeline.")


            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            if not data_transformation_artifact.is_transformed:
                raise Exception("Data Transformation failed. Halting pipeline.")

            # EDA runs on transformed data
            self.start_eda_analysis(data_transformation_artifact) 
            
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)
            if not model_trainer_artifact.is_trained:
                raise Exception("Model Training failed. Halting pipeline.")
            
            clustering_evaluation_artifact = self.start_clustering_evaluation(model_trainer_artifact, data_transformation_artifact)
            if not clustering_evaluation_artifact.is_evaluated:
                 # This might be a warning rather than a hard stop if metrics couldn't be calculated for e.g. DBSCAN
                logging.warning("Clustering Evaluation encountered issues (e.g., metrics not calculable for all models).")
            
            logging.info(f"Training pipeline completed successfully for run: {self.run_timestamp}")

        except Exception as e:
            logging.error(f"Training pipeline failed for run {self.run_timestamp}: {e}")
            logging.error(traceback.format_exc()) # Log full traceback
            # Re-raise if this is run in a context that needs to catch it
            # raise CustomException(e, sys) from e


if __name__ == '__main__':
    # This creates a new timestamped run each time it's executed
    pipeline_timestamp = TIMESTAMP 
    logging.info(f"--- Starting New Training Pipeline Run: {pipeline_timestamp} ---")
    
    # For testing, ensure the source data file exists for DataIngestion
    # This path needs to be correct relative to where this script is run from if not absolute.
    # PROJECT_ROOT_DIR and DATA_FILE_NAME are from constants.

    PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust as needed
    CUSTOMER_DATA_PARENT_DIR = Path(PROJECT_ROOT_DIR) / "customer_data"  # Adjust as needed
    NUMERICAL_COLUMNS = [ # Updated based on CSV inspection, excluding CUST_ID
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
    ]

    original_data_path_for_test = Path(PROJECT_ROOT_DIR) / "customer_data" / "customer_data.csv"
    target_customer_data_path = Path(CUSTOMER_DATA_PARENT_DIR) / "customer_data.csv" 

    if not target_customer_data_path.exists():
        if original_data_path_for_test.exists():
            os.makedirs(target_customer_data_path.parent, exist_ok=True)
            import shutil
            shutil.copy(original_data_path_for_test, target_customer_data_path)
            logging.info(f"Copied {original_data_path_for_test} to {target_customer_data_path} for pipeline run.")
        else:
            # Create a minimal dummy CSV if no source data is found, for the pipeline to run through
            logging.warning(f"No source data found at {original_data_path_for_test} or {target_customer_data_path}. Creating dummy data.")
            os.makedirs(target_customer_data_path.parent, exist_ok=True)
            cols_for_dummy = ["CUST_ID"] + NUMERICAL_COLUMNS
            dummy_data_rows = [
                ["C_TEST1"] + [np.random.rand()*100 for _ in NUMERICAL_COLUMNS],
                ["C_TEST2"] + [np.random.rand()*100 for _ in NUMERICAL_COLUMNS]
            ]
            dummy_df = pd.DataFrame(dummy_data_rows, columns=cols_for_dummy)
            dummy_df.to_csv(target_customer_data_path, index=False)
            logging.info(f"Created dummy data at {target_customer_data_path}.")


    training_pipeline = TrainingPipeline(run_timestamp=pipeline_timestamp)
    try:
        training_pipeline.run_pipeline()
        logging.info(f"--- Training Pipeline Run {pipeline_timestamp} Finished Successfully ---")
    except Exception as e:
        logging.error(f"--- Training Pipeline Run {pipeline_timestamp} FAILED ---")
        # The exception is already logged by run_pipeline, this is just a final marker.

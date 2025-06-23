import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import traceback
import numpy as np

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.config_entity import ModelTrainerConfig
from custsegments.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from custsegments.utils.main_utils.utils import save_object, read_csv, create_directories
# Constants might be needed for default model parameters if not fully in config
from custsegments.constant.training_pipeline import KMEANS_N_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES

class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.transformation_artifact = data_transformation_artifact
        self.config = model_trainer_config
        create_directories([self.config.root_dir, Path(os.path.dirname(self.config.model_save_path))])

    def train_kmeans_model(self, X_train: pd.DataFrame) -> KMeans:
        """
        Trains a KMeans clustering model.
        
        Args:
            X_train (pd.DataFrame): The training data (features).

        Returns:
            KMeans: The trained KMeans model object.
        """
        try:
            logging.info("Starting KMeans model training.")
            kmeans = KMeans(
                n_clusters=self.config.n_clusters_kmeans, 
                init='k-means++', 
                n_init='auto', # Suppresses warning by explicitly setting n_init
                random_state=42
            )
            kmeans.fit(X_train)
            logging.info(f"KMeans model trained with {self.config.n_clusters_kmeans} clusters.")
            logging.info(f"KMeans cluster centers:\n{kmeans.cluster_centers_}") # Can be verbose
            logging.info(f"KMeans inertia: {kmeans.inertia_}")
            return kmeans
        except Exception as e:
            logging.error(f"Error training KMeans model: {e}")
            raise CustomException(e, sys) from e

    def train_dbscan_model(self, X_train: pd.DataFrame) -> DBSCAN:
        """
        Trains a DBSCAN clustering model.
        
        Args:
            X_train (pd.DataFrame): The training data (features).

        Returns:
            DBSCAN: The trained DBSCAN model object.
        """
        try:
            logging.info("Starting DBSCAN model training.")
            dbscan = DBSCAN(
                eps=self.config.eps_dbscan, 
                min_samples=self.config.min_samples_dbscan,
                n_jobs=-1 # Use all available cores
            )
            dbscan.fit(X_train) # DBSCAN doesn't have a predict method in the same way, it assigns labels during fit.
            
            # Log information about the clustering results
            labels = dbscan.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            logging.info(f"DBSCAN model trained. Estimated number of clusters: {n_clusters_}")
            logging.info(f"DBSCAN estimated number of noise points: {n_noise_}")
            # logging.info(f"DBSCAN core sample indices shape: {dbscan.core_sample_indices_.shape}") # Can be verbose
            return dbscan
        except Exception as e:
            logging.error(f"Error training DBSCAN model: {e}")
            raise CustomException(e, sys) from e

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Orchestrates the model training process for both KMeans and DBSCAN.
        Currently, it trains KMeans and saves it as the primary model.
        DBSCAN is trained but not saved by default as per current plan (model.pkl is KMeans).
        This can be adjusted if both models need to be saved or one selected based on pre-evaluation.
        """
        logging.info("Starting model training process.")
        try:
            if not self.transformation_artifact.is_transformed or not self.transformation_artifact.transformed_data_path:
                message = "Data transformation was not successful or path is missing. Skipping model training."
                logging.warning(message)
                return ModelTrainerArtifact(trained_model_path=None, is_trained=False, message=message)

            transformed_data_path = self.transformation_artifact.transformed_data_path
            train_df = read_csv(transformed_data_path)
            logging.info(f"Loaded transformed data for training from: {transformed_data_path}")

            # Assuming all columns in transformed_df are features for clustering
            # (as CUST_ID should have been dropped)
            X_train = train_df 

            # Train KMeans (primary model as per requirements for model.pkl)
            kmeans_model = self.train_kmeans_model(X_train)
            
            # Save the primary model (KMeans)
            # The config.model_save_path should point to where 'model.pkl' (or specific name) is saved.
            # If config.model_name is "model.pkl", it will be saved as such.
            # If config.model_name is "kmeans_model.pkl", it will be saved as such.
            # The requirement is "model.pkl", let's ensure config reflects that or we adjust here.
            
            primary_model_path = self.config.model_save_path # This path should include the filename
            
            # If model_save_path is a directory, append model_name from config
            if os.path.isdir(self.config.model_save_path):
                 primary_model_path = self.config.model_save_path / self.config.model_name
            
            save_object(file_path=primary_model_path, obj=kmeans_model)
            logging.info(f"KMeans model saved to: {primary_model_path}")

            # Train DBSCAN (optional to save, not specified as the primary "model.pkl")
            # dbscan_model = self.train_dbscan_model(X_train)
            # If DBSCAN also needs to be saved, a separate path/name would be needed in config & artifact.
            # For now, we train it, but only KMeans is the "official" saved model.
            # To make it part of the artifact, it needs to be saved and its path returned.
            # Let's assume for now we only need to save the primary model (KMeans).

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=primary_model_path, # Path to the saved KMeans model
                is_trained=True,
                message="Model training (KMeans primary) completed successfully."
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error during model training initiation: {e}\n{traceback.format_exc()}")
            # Create an artifact indicating failure
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=None, 
                is_trained=False, 
                message=f"Model training failed: {str(e)}"
            )
            raise CustomException(e, sys) from e


if __name__ == '__main__':
    from custsegments.constant.training_pipeline import (
        ARTIFACTS_DIR, PIPELINE_NAME, 
        DATA_TRANSFORMATION_DIR_NAME, TRANSFORMED_DATA_FILE, PREPROCESSOR_OBJECT_FILE_NAME,
        MODEL_TRAINER_DIR_NAME, MODEL_OBJECT_FILE_NAME, # Using generic model.pkl
        KMEANS_N_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES # Default params from constants
    )
    import shutil
    
    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent.parent 

    test_artifacts_root = project_root_for_test / ARTIFACTS_DIR / f"{PIPELINE_NAME}_test_mt"
    if test_artifacts_root.exists():
        shutil.rmtree(test_artifacts_root)
    os.makedirs(test_artifacts_root, exist_ok=True)

    # --- Dummy DataTransformationArtifact ---
    dummy_transformed_data_dir = test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME
    os.makedirs(dummy_transformed_data_dir, exist_ok=True)
    
    dummy_transformed_data_path = dummy_transformed_data_dir / TRANSFORMED_DATA_FILE
    dummy_preprocessor_path = dummy_transformed_data_dir / PREPROCESSOR_OBJECT_FILE_NAME # Not used by trainer directly
    
    # Create a dummy transformed CSV for training
    # Using a subset of NUMERICAL_COLUMNS for simplicity
    data_cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS', 'TENURE']
    dummy_data = {col: np.random.rand(100) for col in data_cols} # Already scaled-like data
    dummy_df_transformed = pd.DataFrame(dummy_data)
    dummy_df_transformed.to_csv(dummy_transformed_data_path, index=False)
    logging.info(f"Created dummy transformed data at {dummy_transformed_data_path} for ModelTrainer test.")

    data_transformation_artifact_test = DataTransformationArtifact(
        transformed_data_path=dummy_transformed_data_path,
        preprocessor_object_path=dummy_preprocessor_path, # Not directly used by this test but part of artifact
        is_transformed=True,
        message="Dummy data transformation artifact for model training test."
    )

    # --- ModelTrainerConfig ---
    # MODEL_OBJECT_FILE_NAME is "model.pkl"
    model_save_dir = test_artifacts_root / MODEL_TRAINER_DIR_NAME 
    
    model_trainer_config_test = ModelTrainerConfig(
        root_dir=model_save_dir,
        transformed_data_path=dummy_transformed_data_path, # Input for trainer
        model_name=MODEL_OBJECT_FILE_NAME, # e.g. "model.pkl"
        model_save_path=model_save_dir / MODEL_OBJECT_FILE_NAME, # Full path for the primary model
        n_clusters_kmeans=KMEANS_N_CLUSTERS, # Use default from constants
        eps_dbscan=DBSCAN_EPS,
        min_samples_dbscan=DBSCAN_MIN_SAMPLES
    )
    logging.info(f"Model Trainer Config for test: {model_trainer_config_test}")

    try:
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact_test,
            model_trainer_config=model_trainer_config_test
        )
        model_trainer_artifact = model_trainer.initiate_model_training()
        logging.info(f"Model Trainer component test finished. Artifact: {model_trainer_artifact}")

        assert model_trainer_artifact.is_trained
        assert model_trainer_artifact.trained_model_path.exists()
        logging.info(f"Trained model (KMeans) saved at: {model_trainer_artifact.trained_model_path}")

        # Additionally, test DBSCAN training (though not saved in this main path)
        X_test_data = read_csv(dummy_transformed_data_path)
        dbscan_test_model = model_trainer.train_dbscan_model(X_test_data)
        assert dbscan_test_model is not None, "DBSCAN model training failed in test."
        logging.info("DBSCAN model trained successfully in test (not saved as primary).")
        
        # Clean up (optional)
        # if test_artifacts_root.exists():
        #    shutil.rmtree(test_artifacts_root)
        #    logging.info(f"Cleaned up test artifacts directory: {test_artifacts_root}")

    except CustomException as ce:
        logging.error(f"ModelTrainer component test failed: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error in ModelTrainer component test: {e}")
        logging.error(traceback.format_exc())

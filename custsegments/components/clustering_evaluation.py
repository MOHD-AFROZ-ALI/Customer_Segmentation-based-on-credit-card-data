import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN # To load model instances if needed, or use type hinting
import matplotlib.pyplot as plt
import seaborn as sns
import json
import traceback

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.config_entity import ClusteringEvaluationConfig
from custsegments.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ClusteringEvaluationArtifact
from custsegments.utils.main_utils.utils import load_object, read_csv, save_json, create_directories,save_object
from custsegments.utils.ml_utils.clustering_metrics import calculate_silhouette_score, calculate_davies_bouldin_score, plot_elbow_method, calculate_elbow_method_sse # Assuming this has the metrics
from custsegments.constant.training_pipeline import NUMERICAL_COLUMNS
class ClusteringEvaluation:
    def __init__(self,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact, # Need original (or pre-scaled) data for some interpretations
                 clustering_evaluation_config: ClusteringEvaluationConfig):
        self.trainer_artifact = model_trainer_artifact
        self.transformation_artifact = data_transformation_artifact
        self.config = clustering_evaluation_config
        create_directories([self.config.root_dir, Path(os.path.dirname(self.config.metrics_file_path))])
        # If plots are saved, ensure directory exists
        # create_directories([self.config.root_dir / "plots"]) # Example path for plots from config

    def get_cluster_labels(self, model: any, data: pd.DataFrame) -> np.ndarray:
        """
        Gets cluster labels from the trained model.
        Handles different ways models might provide labels (e.g., .labels_ vs .predict()).
        """
        if hasattr(model, 'labels_') and not callable(model.labels_): # DBSCAN stores labels directly
            return model.labels_
        elif hasattr(model, 'predict') and callable(model.predict): # KMeans uses predict
            return model.predict(data)
        else:
            raise ValueError("Model does not have a standard way to get cluster labels ('labels_' attribute or 'predict' method).")

    def evaluate_model_performance(self, model: any, data: pd.DataFrame, model_name: str) -> dict:
        """
        Evaluates a single clustering model using Silhouette Score and Davies-Bouldin Index.
        
        Args:
            model: Trained clustering model object.
            data (pd.DataFrame): Data used for clustering (usually scaled).
            model_name (str): Name of the model for logging/reporting.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        try:
            logging.info(f"Evaluating {model_name}...")
            labels = self.get_cluster_labels(model, data)
            
            metrics = {}
            # Silhouette Score (higher is better, range -1 to 1)
            # Requires at least 2 clusters (excluding noise if DBSCAN)
            unique_labels = np.unique(labels)
            n_clusters_for_metric = len(unique_labels) - (1 if -1 in unique_labels else 0)

            if n_clusters_for_metric >= 2:
                # For metrics, it's common to exclude noise points for DBSCAN
                if -1 in unique_labels and isinstance(model, DBSCAN):
                    core_samples_mask = (labels != -1)
                    if np.sum(core_samples_mask) > 1 and len(np.unique(labels[core_samples_mask])) >=2 : # Check if there are enough non-noise points and clusters
                        metrics['silhouette_score'] = calculate_silhouette_score(data[core_samples_mask], labels[core_samples_mask])
                        metrics['davies_bouldin_score'] = calculate_davies_bouldin_score(data[core_samples_mask], labels[core_samples_mask])
                    else:
                        logging.warning(f"Not enough non-noise samples or clusters for {model_name} to calculate metrics.")
                        metrics['silhouette_score'] = np.nan
                        metrics['davies_bouldin_score'] = np.nan
                else: # For KMeans or DBSCAN without noise points / or if we evaluate on all points
                    metrics['silhouette_score'] = calculate_silhouette_score(data, labels)
                    metrics['davies_bouldin_score'] = calculate_davies_bouldin_score(data, labels)
            else:
                logging.warning(f"{model_name} resulted in < 2 actual clusters. Metrics cannot be calculated.")
                metrics['silhouette_score'] = np.nan 
                metrics['davies_bouldin_score'] = np.nan
            
            metrics['num_clusters_found'] = n_clusters_for_metric
            if isinstance(model, DBSCAN):
                metrics['noise_points_found'] = list(labels).count(-1)

            logging.info(f"{model_name} metrics: {metrics}")
            return metrics

        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {e}")
            raise CustomException(e, sys) from e
            
    def perform_elbow_method_for_kmeans(self, data: pd.DataFrame, max_k: int = 10) -> dict:
        """
        Performs the Elbow method to help find optimal k for KMeans.
        Saves a plot of the SSE vs. Number of clusters.
        """
        try:
            logging.info(f"Performing Elbow method for KMeans (max_k={max_k})...")
            sse_values = calculate_elbow_method_sse(data, max_k=max_k)
            
            # Plotting
            plot_save_path = Path(self.config.root_dir) / "elbow_method_plot.png"
            plot_elbow_method(sse_values, save_path=plot_save_path)
            logging.info(f"Elbow method plot saved to {plot_save_path}")
            return sse_values
        except Exception as e:
            logging.error(f"Error during Elbow method: {e}")
            raise CustomException(e, sys) from e

    def extract_business_insights(self, data_original_scale: pd.DataFrame, labels: np.ndarray, model_name: str) -> str:
        """
        Analyzes clusters to extract business insights.
        This usually involves looking at the centroids or characteristics of each cluster
        on the original data scale.

        Args:
            data_original_scale (pd.DataFrame): Dataframe with features on their original scale.
            labels (np.ndarray): Cluster labels for each data point.
            model_name (str): Name of the model for context.
        
        Returns:
            str: A string containing textual business insights.
        """
        logging.info(f"Extracting business insights for {model_name}...")
        insights_text = f"Business Insights from {model_name} Clusters:\n"
        insights_text += "==============================================\n"

        # Add original labels to the dataframe for easier analysis
        df_labeled = data_original_scale.copy()
        df_labeled['CLUSTER'] = labels

        # Exclude noise points if any (common for DBSCAN insights)
        if -1 in np.unique(labels):
            insights_text += f"Note: Noise points (CLUSTER == -1) are excluded from this summary.\n"
            df_labeled_no_noise = df_labeled[df_labeled['CLUSTER'] != -1]
        else:
            df_labeled_no_noise = df_labeled
        
        if df_labeled_no_noise.empty or df_labeled_no_noise['CLUSTER'].nunique() == 0:
            insights_text += "No actual clusters found to analyze (excluding noise).\n"
            logging.warning("No actual clusters to analyze for business insights.")
            return insights_text

        # Calculate mean/median of features for each cluster
        cluster_summary = df_labeled_no_noise.groupby('CLUSTER').mean() # Could use .median() too
        
        insights_text += "\nCluster Profiles (Mean Values of Features):\n"
        insights_text += cluster_summary.to_string()
        insights_text += "\n\nInterpretation Hints:\n"

        for cluster_id in sorted(df_labeled_no_noise['CLUSTER'].unique()):
            insights_text += f"\n--- Cluster {cluster_id} ---\n"
            insights_text += f"   - Size: {np.sum(df_labeled_no_noise['CLUSTER'] == cluster_id)} customers\n"
            # Example: Highlight features that are significantly higher or lower than average for this cluster
            for col in cluster_summary.columns:
                cluster_mean = cluster_summary.loc[cluster_id, col]
                overall_mean = data_original_scale[col].mean() # Compare to overall mean
                if cluster_mean > overall_mean * 1.2: # 20% higher
                    insights_text += f"   - High {col}: {cluster_mean:.2f} (Overall avg: {overall_mean:.2f})\n"
                elif cluster_mean < overall_mean * 0.8: # 20% lower
                    insights_text += f"   - Low {col}: {cluster_mean:.2f} (Overall avg: {overall_mean:.2f})\n"
            insights_text += "   - Potential Segment Name: [e.g., High Spenders, Low Balance Holders, etc. - requires domain knowledge]\n"
        
        # Save insights to a file (optional, could be part of main report)
        insights_file_path = Path(self.config.root_dir) / f"{model_name}_insights.txt"
        with open(insights_file_path, "w") as f:
           f.write(insights_text)
        logging.info(f"Business insights for {model_name} saved to {insights_file_path}")
        
        return insights_text


    def initiate_clustering_evaluation(self) -> ClusteringEvaluationArtifact:
        """
        Main method to evaluate clustering models.
        """
        logging.info("Starting clustering model evaluation.")
        try:
            if not self.trainer_artifact.is_trained or not self.trainer_artifact.trained_model_path:
                message = "Model training was not successful or model path is missing. Skipping evaluation."
                logging.warning(message)
                return ClusteringEvaluationArtifact(evaluation_metrics={}, is_evaluated=False, message=message)

            # Load the primary trained model (KMeans as per current plan)
            primary_model = load_object(self.trainer_artifact.trained_model_path)
            logging.info(f"Loaded primary model from: {self.trainer_artifact.trained_model_path}")

            # Load the transformed data (used for training and for metrics calculation)
            transformed_data = read_csv(self.transformation_artifact.transformed_data_path)
            logging.info(f"Loaded transformed data for evaluation. Shape: {transformed_data.shape}")

            # --- KMeans Evaluation ---
            kmeans_metrics = self.evaluate_model_performance(primary_model, transformed_data, "KMeans")
            # Perform Elbow method for KMeans (if not already done or for reporting)
            self.perform_elbow_method_for_kmeans(transformed_data) # This saves a plot

            # --- DBSCAN Evaluation (Requires DBSCAN model to be trained and accessible) ---
            # For this, we'd need to load/train DBSCAN.
            # Assuming ModelTrainer trains both and we can access DBSCAN parameters from config.
            # For simplicity, let's re-train DBSCAN here if its path is not in trainer_artifact.
            # This is not ideal for production (should be trained once in trainer).
            # Alternative: ModelTrainerArtifact could include paths to ALL trained models.
            
            logging.info("Re-training DBSCAN for evaluation purposes (ideally, load if saved by trainer).")
            # This assumes ModelTrainerConfig is accessible or its relevant parts are in ClusteringEvaluationConfig
            dbscan_model_eval = DBSCAN(
                #eps=self.config.eps_dbscan if hasattr(self.config, 'eps_dbscan') else 0.5, # Fallback defaults
                #min_samples=self.config.min_samples_dbscan if hasattr(self.config, 'min_samples_dbscan') else 5,
                eps=0.5, # Example value, should be from config
                min_samples=5, # Example value, should be from config
                n_jobs=-1
            )
            dbscan_model_eval.fit(transformed_data)
            dbscan_metrics = self.evaluate_model_performance(dbscan_model_eval, transformed_data, "DBSCAN")

            all_metrics = {
                "KMeans": kmeans_metrics,
                "DBSCAN": dbscan_metrics
            }
            save_json(self.config.metrics_file_path, all_metrics)
            logging.info(f"Clustering metrics saved to: {self.config.metrics_file_path}")

            # --- Business Insights ---
            # For business insights, we need data on the original scale.
            # This requires access to data *before* scaling, but *after* imputation and CUST_ID drop.
            # The current DataTransformationArtifact only gives the fully scaled data.
            # This highlights a potential need for an intermediate artifact from DataTransformation
            # or loading the preprocessor and inverse_transforming.


            # we should use inverse_transform on the scaled data to get back to original scale.And then generate insights in real scenario. 

            # # Simplified approach: use the preprocessor to inverse_transform the scaled data.
            # preprocessor = load_object(self.transformation_artifact.preprocessor_object_path)
            
            # # Get columns that were scaled (NUMERICAL_COLUMNS after dropping CUST_ID)
            # numerical_cols_for_inverse = [col for col in NUMERICAL_COLUMNS if col in transformed_data.columns]

            # if numerical_cols_for_inverse:
            #     try:
            #         data_original_scale_array = preprocessor.inverse_transform(transformed_data[numerical_cols_for_inverse])
            #     except Exception as e:
            #         logging.error(f"Error during inverse transformation: {e}")
            #         raise CustomException(e, sys) from e
            #     #data_original_scale_array = preprocessor.inverse_transform(transformed_data[numerical_cols_for_inverse])
            #     data_original_scale_df = pd.DataFrame(data_original_scale_array, columns=numerical_cols_for_inverse, index=transformed_data.index)

        

            # # Add back any non-numerical columns if they were part of transformed_data (should not be, but for safety)
            #     for col in transformed_data.columns:
            #         if col not in data_original_scale_df.columns:
            #             data_original_scale_df[col] = transformed_data[col]

            #     logging.info("Inverse transformed data for business insights.")

            # For simplicity, we are taking scaled data as original scale for insights.
            data_for_insights = transformed_data
            kmeans_labels = self.get_cluster_labels(primary_model, data_for_insights)
            kmeans_insights = self.extract_business_insights(data_for_insights, kmeans_labels, "KMeans")

            dbscan_labels = self.get_cluster_labels(dbscan_model_eval, data_for_insights)
            dbscan_insights = self.extract_business_insights(data_for_insights, dbscan_labels, "DBSCAN")
            
            full_report_content = f"CLUSTERING EVALUATION & INSIGHTS REPORT\n"
            full_report_content += "=========================================\n\n"
            full_report_content += f"Metrics File: {self.config.metrics_file_path}\n"
            full_report_content += f"Metrics Content:\n{json.dumps(all_metrics, indent=2)}\n\n"
            full_report_content += kmeans_insights
            full_report_content += "\n\n" + ("-"*50) + "\n\n"
            full_report_content += dbscan_insights

            # Save combined report (example path)
            insights_file_path = self.config.insights_file_path # if defined in config
            insights_file_path = Path(self.config.root_dir) / "clustering_insights_report.txt"
            with open(insights_file_path, "w") as f:
                f.write(full_report_content)
            logging.info(f"Combined evaluation and insights report saved to {insights_file_path}")

            # else:
            #     logging.warning("No numerical columns found to inverse transform for business insights.")


            evaluation_artifact = ClusteringEvaluationArtifact(
                evaluation_metrics=all_metrics,
                insights_path=insights_file_path, # If saving insights separately and want to track its path
                is_evaluated=True,
                message="Clustering evaluation and initial business insights extraction completed."
            )
            logging.info(f"Clustering evaluation artifact: {evaluation_artifact}")
            return evaluation_artifact

        except Exception as e:
            logging.error(f"Error during clustering evaluation: {e}\n{traceback.format_exc()}")
            # Create an artifact indicating failure
            evaluation_artifact = ClusteringEvaluationArtifact(
                evaluation_metrics={}, 
                is_evaluated=False, 
                message=f"Clustering evaluation failed: {str(e)}"
            )
            raise CustomException(e, sys) from e


if __name__ == '__main__':
    from custsegments.constant.training_pipeline import (
        ARTIFACTS_DIR, PIPELINE_NAME, NUMERICAL_COLUMNS,
        MODEL_TRAINER_DIR_NAME, MODEL_OBJECT_FILE_NAME,
        DATA_TRANSFORMATION_DIR_NAME, TRANSFORMED_DATA_FILE, PREPROCESSOR_OBJECT_FILE_NAME,
        CLUSTERING_EVALUATION_DIR_NAME, MODEL_EVALUATION_METRICS_FILE
    )
    from sklearn.pipeline import Pipeline # For dummy preprocessor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import shutil

    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent.parent

    test_artifacts_root = project_root_for_test / ARTIFACTS_DIR / f"{PIPELINE_NAME}_test_eval"
    if test_artifacts_root.exists():
        shutil.rmtree(test_artifacts_root)
    os.makedirs(test_artifacts_root, exist_ok=True)

    # --- Dummy ModelTrainerArtifact ---
    dummy_model_trainer_dir = test_artifacts_root / MODEL_TRAINER_DIR_NAME
    os.makedirs(dummy_model_trainer_dir, exist_ok=True)
    dummy_model_path = dummy_model_trainer_dir / MODEL_OBJECT_FILE_NAME
    
    # Create and save a dummy KMeans model
    dummy_kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
    # Need some dummy data to fit it, use a small subset of NUMERICAL_COLUMNS
    fit_cols = NUMERICAL_COLUMNS[:3] if len(NUMERICAL_COLUMNS) >=3 else NUMERICAL_COLUMNS
    dummy_fit_data = pd.DataFrame(np.random.rand(10, len(fit_cols)), columns=fit_cols)
    dummy_kmeans.fit(dummy_fit_data) # Fit with some data
    save_object(dummy_model_path, dummy_kmeans)
    
    model_trainer_artifact_test = ModelTrainerArtifact(
        trained_model_path=dummy_model_path,
        is_trained=True,
        message="Dummy model trainer artifact for evaluation test."
    )

    # --- Dummy DataTransformationArtifact ---
    dummy_transformed_data_dir = test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME
    os.makedirs(dummy_transformed_data_dir, exist_ok=True)
    dummy_transformed_data_path = dummy_transformed_data_dir / TRANSFORMED_DATA_FILE
    dummy_preprocessor_path = dummy_transformed_data_dir / PREPROCESSOR_OBJECT_FILE_NAME

    # Create dummy transformed data (scaled)
    data_for_eval = {col: np.random.randn(100) for col in NUMERICAL_COLUMNS} # Scaled-like
    dummy_df_for_eval = pd.DataFrame(data_for_eval)
    dummy_df_for_eval.to_csv(dummy_transformed_data_path, index=False)
    
    # Create and save dummy preprocessor
    dummy_preprocessor = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
    # Fit preprocessor on some data so it's "fitted"
    dummy_preprocessor_fit_data = pd.DataFrame(np.random.rand(10, len(NUMERICAL_COLUMNS)), columns=NUMERICAL_COLUMNS)
    dummy_preprocessor.fit(dummy_preprocessor_fit_data)
    save_object(dummy_preprocessor_path, dummy_preprocessor)

    data_transformation_artifact_test = DataTransformationArtifact(
        transformed_data_path=dummy_transformed_data_path,
        preprocessor_object_path=dummy_preprocessor_path,
        is_transformed=True,
        message="Dummy data transformation for evaluation test."
    )
    
    # --- ClusteringEvaluationConfig ---
    eval_output_dir = test_artifacts_root / CLUSTERING_EVALUATION_DIR_NAME
    
    # Use dummy params for DBSCAN from ModelTrainerConfig for consistency
    # These would normally come from a shared config or tuning step.
    # For test, directly use some values.
    DBSCAN_EPS_TEST = 0.5 
    DBSCAN_MIN_SAMPLES_TEST = 5

    clustering_evaluation_config_test = ClusteringEvaluationConfig(
        root_dir=eval_output_dir,
        # These paths are relative to where the actual model and data would be
        # For testing, we use the dummy artifact paths directly
        model_path=model_trainer_artifact_test.trained_model_path, 
        preprocessor_path=data_transformation_artifact_test.preprocessor_object_path,
        data_path=data_transformation_artifact_test.transformed_data_path, # This is the scaled data
        metrics_file_path=eval_output_dir / MODEL_EVALUATION_METRICS_FILE,
        # Adding DBSCAN params to config for evaluation component to use
        eps_dbscan=DBSCAN_EPS_TEST,
        min_samples_dbscan=DBSCAN_MIN_SAMPLES_TEST
    )
    # Also need to ensure the config entity has these fields:
    # In entity/config_entity.py, ClusteringEvaluationConfig should have eps_dbscan and min_samples_dbscan
    # For now, this test will work if they are attributes of the config object.

    logging.info(f"Clustering Evaluation Config for test: {clustering_evaluation_config_test}")

    try:
        clustering_evaluation = ClusteringEvaluation(
            model_trainer_artifact=model_trainer_artifact_test,
            data_transformation_artifact=data_transformation_artifact_test,
            clustering_evaluation_config=clustering_evaluation_config_test
        )
        evaluation_artifact = clustering_evaluation.initiate_clustering_evaluation()
        logging.info(f"Clustering Evaluation component test finished. Artifact: {evaluation_artifact}")

        assert evaluation_artifact.is_evaluated
        assert evaluation_artifact.evaluation_metrics is not None
        assert "KMeans" in evaluation_artifact.evaluation_metrics
        assert "DBSCAN" in evaluation_artifact.evaluation_metrics
        assert Path(clustering_evaluation_config_test.metrics_file_path).exists()
        assert (Path(eval_output_dir) / "elbow_method_plot.png").exists()
        assert (Path(eval_output_dir) / "clustering_insights_report.txt").exists()
        
        logging.info(f"Metrics saved at: {clustering_evaluation_config_test.metrics_file_path}")
        logging.info(f"Evaluation metrics content: {json.dumps(evaluation_artifact.evaluation_metrics, indent=2)}")
        
        # Clean up (optional)
        # if test_artifacts_root.exists():
        #    shutil.rmtree(test_artifacts_root)
        #    logging.info(f"Cleaned up test artifacts directory: {test_artifacts_root}")

    except CustomException as ce:
        logging.error(f"ClusteringEvaluation component test failed: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error in ClusteringEvaluation component test: {e}")
        logging.error(traceback.format_exc())
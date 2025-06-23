from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_url: str # Not used in this project as data is local
    local_data_file: Path
    unzip_dir: Path # Not used in this project

@dataclass
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    validation_report_path: Path
    all_required_files: list
    data_path: Path # Path to the ingested data for validation
    schema: dict # Define expected schema

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path # Path to validated data
    preprocessor_obj_file_path: Path
    transformed_data_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    transformed_data_path: Path
    model_name: str # e.g., "kmeans_model.pkl"
    model_save_path: Path
    # Add Kmeans specific params
    n_clusters_kmeans: int
    # Add DBSCAN specific params
    eps_dbscan: float
    min_samples_dbscan: int

@dataclass
class ClusteringEvaluationConfig:
    root_dir: Path
    model_path: Path
    preprocessor_path: Path
    data_path: Path # Path to original or transformed data for evaluation with labels
    metrics_file_path: Path
    #eps_dbscan: float
    #min_samples_dbscan: int
    # Add params for report generation if any, e.g., plots directory
    insights_file_path: Path # For saving textual insights


@dataclass
class TrainingPipelineConfig:
    artifacts_root: Path

@dataclass
class BatchPredictionConfig:
    input_csv_path: Path
    output_csv_path: Path
    model_path: Path
    preprocessor_path: Path
    # Define how to map cluster labels to business insights if needed
    # cluster_insights_map: dict

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    ingested_data_path: Path
    is_ingested: bool
    message: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    validated_data_path: Path # Path to the data if validation is successful
    validation_report_path: Path # Path to a report detailing validation checks
    message: str

@dataclass
class DataTransformationArtifact:
    transformed_data_path: Path
    preprocessor_object_path: Path
    #imputed_data_path: Path # Path to data after imputation
    is_transformed: bool
    message: str

@dataclass
class ModelTrainerArtifact:
    trained_model_path: Path # Path to the primary model (e.g., KMeans)
    # trained_dbscan_model_path: Path # Optional: if saving DBSCAN separately
    is_trained: bool
    message: str

@dataclass
class ClusteringEvaluationArtifact:
    evaluation_metrics: dict # e.g., {'silhouette': 0.5, 'davies_bouldin': 1.2}
    # evaluation_report_path: Path # Path to a more detailed report or visualizations
    insights_path: Path # Path to saved business insights
    is_evaluated: bool
    message: str

@dataclass
class BatchPredictionArtifact:
    output_file_path: Path
    num_records_processed: int
    is_prediction_successful: bool
    message: str

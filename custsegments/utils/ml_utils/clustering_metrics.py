import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.utils.main_utils.utils import save_json # For saving metrics if needed


def calculate_silhouette_score(X: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Calculates the Silhouette Score for clustering.

    Args:
        X (pd.DataFrame): Data used for clustering (features).
        labels (np.ndarray): Cluster labels for each point.

    Returns:
        float: Silhouette Score.
    
    Raises:
        CustomException: If an error occurs during calculation.
    """
    try:
        if len(np.unique(labels)) < 2: # Silhouette score is not defined for a single cluster
            logging.warning("Silhouette score cannot be calculated with less than 2 clusters. Returning -1.")
            return -1.0 
        score = silhouette_score(X, labels)
        logging.info(f"Calculated Silhouette Score: {score}")
        return score
    except Exception as e:
        raise CustomException(e, sys) from e

def calculate_davies_bouldin_score(X: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Calculates the Davies-Bouldin Index for clustering.

    Args:
        X (pd.DataFrame): Data used for clustering (features).
        labels (np.ndarray): Cluster labels for each point.

    Returns:
        float: Davies-Bouldin Index.
        
    Raises:
        CustomException: If an error occurs during calculation.
    """
    try:
        if len(np.unique(labels)) < 2: # Davies-Bouldin score is not defined for a single cluster
            logging.warning("Davies-Bouldin score cannot be calculated with less than 2 clusters. Returning -1.")
            return -1.0 # Or a large positive number, as lower is better
        score = davies_bouldin_score(X, labels)
        logging.info(f"Calculated Davies-Bouldin Score: {score}")
        return score
    except Exception as e:
        raise CustomException(e, sys) from e

def calculate_elbow_method_sse(X: pd.DataFrame, max_k: int = 10, random_state: int = 42) -> dict:
    """
    Calculates Sum of Squared Errors (SSE) for different values of k for KMeans (Elbow Method).

    Args:
        X (pd.DataFrame): Data to cluster.
        max_k (int, optional): Maximum number of clusters to test. Defaults to 10.
        random_state (int, optional): Random state for KMeans. Defaults to 42.

    Returns:
        dict: A dictionary where keys are k values and values are corresponding SSE.
        
    Raises:
        CustomException: If an error occurs during calculation.
    """
    sse = {}
    try:
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state, n_init='auto')
            kmeans.fit(X)
            sse[k] = kmeans.inertia_  # Inertia is the SSE
        logging.info(f"Calculated SSE for Elbow Method (up to k={max_k}): {sse}")
        return sse
    except Exception as e:
        raise CustomException(e, sys) from e

def plot_elbow_method(sse_values: dict, save_path: Path = None):
    """
    Plots the Elbow Method graph.

    Args:
        sse_values (dict): Dictionary of k and SSE values from calculate_elbow_method_sse.
        save_path (Path, optional): Path to save the plot. If None, plot is shown.
    
    Raises:
        CustomException: If an error occurs during plotting.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(list(sse_values.keys()), list(sse_values.values()), marker='o', linestyle='--')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title("Elbow Method for Optimal k")
        plt.xticks(list(sse_values.keys()))
        plt.grid(True)
        if save_path:
            # Ensure parent directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Elbow method plot saved to {save_path}")
        else:
            plt.show()
        plt.close() # Close the plot figure to free memory
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_clustering_metrics(X: pd.DataFrame, labels: np.ndarray, model_name: str) -> dict:
    """
    Calculates and returns multiple clustering metrics.

    Args:
        X (pd.DataFrame): Feature set.
        labels (np.ndarray): Cluster labels.
        model_name (str): Name of the clustering model (e.g., "KMeans", "DBSCAN").

    Returns:
        dict: Dictionary containing Silhouette Score and Davies-Bouldin Index.
              Returns empty dict if metrics can't be calculated (e.g. < 2 clusters).
    """
    metrics = {}
    num_clusters = len(np.unique(labels))
    num_noise_points = np.sum(labels == -1) if -1 in labels else 0
    
    logging.info(f"Evaluating {model_name}: Found {num_clusters} clusters (including noise if present). Noise points: {num_noise_points}")

    if num_clusters < 2 or (num_clusters == 2 and -1 in labels and num_noise_points > 0 and (len(labels) - num_noise_points) < 2): # check if actual clusters < 2
        logging.warning(f"{model_name}: Not enough clusters ({num_clusters} including noise) to calculate Silhouette or Davies-Bouldin. Skipping.")
        return metrics # Return empty or specific indication

    # For DBSCAN, it's common to evaluate on non-noise points
    if -1 in labels:
        core_samples_mask = (labels != -1)
        X_core = X[core_samples_mask]
        labels_core = labels[core_samples_mask]
        if len(np.unique(labels_core)) < 2:
             logging.warning(f"{model_name}: Not enough actual clusters (excluding noise) to calculate Silhouette or Davies-Bouldin. Skipping.")
             return metrics
    else:
        X_core = X
        labels_core = labels

    metrics['silhouette_score'] = calculate_silhouette_score(X_core, labels_core)
    metrics['davies_bouldin_score'] = calculate_davies_bouldin_score(X_core, labels_core)
    metrics['num_clusters'] = num_clusters - (1 if -1 in labels else 0) # Number of actual clusters
    metrics['noise_points'] = num_noise_points
    
    logging.info(f"{model_name} Evaluation Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    # Example Usage (requires some sample data)
    from sklearn.datasets import make_blobs

    # Generate sample data
    X_sample, y_true_sample = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    X_sample_df = pd.DataFrame(X_sample, columns=['feature1', 'feature2'])

    # --- Test Elbow Method ---
    try:
        logging.info("--- Testing Elbow Method ---")
        # Scale data for KMeans
        scaler = StandardScaler()
        X_scaled_sample = scaler.fit_transform(X_sample_df)
        
        sse = calculate_elbow_method_sse(X_scaled_sample, max_k=8)
        # Create a dummy save path for the plot
        # Ensure the path is relative to where this script might be run from, or use absolute paths for tests
        current_file_dir = Path(__file__).parent
        test_plots_dir = current_file_dir / "test_plots"
        test_plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_elbow_method(sse, save_path=test_plots_dir / "elbow_test_plot.png")
        logging.info(f"Elbow method SSE: {sse}")

        # --- Test KMeans Evaluation ---
        logging.info("\n--- Testing KMeans Evaluation ---")
        optimal_k = 4 # From visual inspection of elbow or prior knowledge for make_blobs
        kmeans_sample = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        kmeans_labels_sample = kmeans_sample.fit_predict(X_scaled_sample)
        
        kmeans_metrics = evaluate_clustering_metrics(X_scaled_sample, kmeans_labels_sample, "KMeans_Sample")
        logging.info(f"KMeans Sample Metrics: {kmeans_metrics}")
        assert 'silhouette_score' in kmeans_metrics
        assert 'davies_bouldin_score' in kmeans_metrics

        # --- Test DBSCAN Evaluation ---
        logging.info("\n--- Testing DBSCAN Evaluation ---")
        # DBSCAN parameters might need tuning for make_blobs
        dbscan_sample = DBSCAN(eps=0.5, min_samples=5) # These params might result in all noise or one cluster
        dbscan_labels_sample = dbscan_sample.fit_predict(X_scaled_sample)

        # Check if DBSCAN found any clusters
        if len(np.unique(dbscan_labels_sample)) > 1 :
            dbscan_metrics = evaluate_clustering_metrics(X_scaled_sample, dbscan_labels_sample, "DBSCAN_Sample")
            logging.info(f"DBSCAN Sample Metrics: {dbscan_metrics}")
            # Assertions might fail if DBSCAN params are not well-tuned for the sample data
            # For example, if all points are noise, or only one cluster is found.
            if dbscan_metrics: # if metrics were calculated
                 assert 'silhouette_score' in dbscan_metrics
                 assert 'davies_bouldin_score' in dbscan_metrics
        else:
            logging.warning(f"DBSCAN found {len(np.unique(dbscan_labels_sample))} unique labels (clusters/noise). Metrics might be empty.")
            # Example of how to handle cases where metrics might not be calculable
            dbscan_metrics = evaluate_clustering_metrics(X_scaled_sample, dbscan_labels_sample, "DBSCAN_Sample")
            logging.info(f"DBSCAN Sample Metrics (potentially empty): {dbscan_metrics}")


        logging.info("\nAll ml_utils tests completed.")

    except CustomException as ce:
        logging.error(f"A custom exception occurred: {ce}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # import shutil
        # if test_plots_dir.exists(): shutil.rmtree(test_plots_dir)
        pass

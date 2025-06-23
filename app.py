import os
import sys
import uuid
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import pandas as pd
import traceback

import numpy as np # Added numpy
# Assuming custsegments is in PYTHONPATH or installed
from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.pipeline.batch_prediction import BatchPredictionPipeline
from custsegments.entity.config_entity import BatchPredictionConfig
from custsegments.utils.main_utils.utils import read_csv # For reading the predicted file to display

# Constants for file paths - these should align with your project structure
# PROJECT_ROOT_DIR will be 'customer_segments/'
# PREDICTION_OUTPUT_DIR will be 'customer_segments/prediction_output/'
# FINAL_MODEL_DIR will be 'customer_segments/final_model/'
from custsegments.constant.training_pipeline import (
    PROJECT_ROOT_DIR, PREDICTION_OUTPUT_DIR, FINAL_MODEL_DIR,
    PREPROCESSOR_OBJECT_FILE_NAME, MODEL_OBJECT_FILE_NAME, BATCH_PREDICTION_OUTPUT_FILE
)

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = "customersegmentationsecretkey" # For flash messages

# Define upload folder relative to the app.py location
# This assumes app.py is in customer_segments/
UPLOAD_FOLDER = Path(PROJECT_ROOT_DIR) / 'uploads' 
# Define prediction output folder also relative to app.py
# This is where BatchPredictionPipeline will save its output, and Flask will read from
# PREDICTION_OUTPUT_DIR is already defined as customer_segments/prediction_output/
PREDICTION_FOLDER = Path(PROJECT_ROOT_DIR) / PREDICTION_OUTPUT_DIR 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

# Path to the trained model and preprocessor
# These are expected to be in customer_segments/final_model/
MODEL_PATH = Path(PROJECT_ROOT_DIR) / FINAL_MODEL_DIR / MODEL_OBJECT_FILE_NAME
PREPROCESSOR_PATH = Path(PROJECT_ROOT_DIR) / FINAL_MODEL_DIR / PREPROCESSOR_OBJECT_FILE_NAME


def generate_summary_insights(df_clustered: pd.DataFrame) -> dict:
    """
    Generates summary insights from the clustered DataFrame.
    Example: Number of customers per cluster.
    """
    insights = {}
    if 'CLUSTER' in df_clustered.columns:
        cluster_counts = df_clustered['CLUSTER'].value_counts().sort_index()
        insights['cluster_distribution'] = cluster_counts.to_dict()
        insights['total_customers'] = int(df_clustered.shape[0])
        insights['num_clusters_found'] = int(df_clustered['CLUSTER'].nunique())
        # More complex insights can be added here by analyzing feature distributions per cluster
        # For example, mean 'PURCHASES' per cluster:
        # insights['mean_purchases_per_cluster'] = df_clustered.groupby('CLUSTER')['PURCHASES'].mean().to_dict() 
        # This requires original features to be present or passed alongside scaled data + cluster labels
    return insights

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html', results=None, insights=None, visuals=None, uploaded_filename=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, triggers batch prediction, and displays results."""
from flask import jsonify # Added jsonify

# ... (other imports remain the same)

# Helper function to render partials to string for AJAX
def render_partial_to_string(template_name, **kwargs):
    return render_template(template_name, **kwargs)

# --- Business Insights Generation ---
def generate_business_recommendations(original_df_with_clusters: pd.DataFrame, numerical_cols: list) -> list:
    """
    Generates business insights and recommendations based on clustered original data.
    Assumes original_df_with_clusters has original feature values and a 'CLUSTER' column.
    """
    recommendations = []

    # Calculate overall means for comparison (optional, can make insights relative to dataset average)
    # overall_means = original_df_with_clusters[numerical_cols].mean()

    for cluster_id in sorted(original_df_with_clusters['CLUSTER'].unique()):
        cluster_data = original_df_with_clusters[original_df_with_clusters['CLUSTER'] == cluster_id]
        cluster_means = cluster_data[numerical_cols].mean()

        rec = {
            'cluster_name': f"Cluster {cluster_id}",
            'description': f"Analysis of Cluster {cluster_id}.",
            'characteristics': [],
            'actions': []
        }

        # --- Heuristic Rules for Insights (examples based on common credit card features) ---
        # These rules are illustrative and would need refinement based on actual data distributions and business goals.
        # Using hardcoded thresholds here for simplicity. A more robust approach would use quantiles or deviations from overall mean.

        # Example: High Balance & High Purchases -> "Premium Customers"
        if 'BALANCE' in cluster_means and 'PURCHASES' in cluster_means:
            if cluster_means['BALANCE'] > 2000 and cluster_means['PURCHASES'] > 1500: # Example thresholds
                rec['cluster_name'] += " - Potential Premium"
                rec['characteristics'].append(f"High Average Balance (${cluster_means['BALANCE']:.2f})")
                rec['characteristics'].append(f"High Average Purchases (${cluster_means['PURCHASES']:.2f})")
                rec['actions'].append("Offer loyalty programs and premium services.")
                rec['actions'].append("Market high-value products/services.")

        # Example: High Credit Limit Usage (Balance close to Credit Limit)
        if 'BALANCE' in cluster_means and 'CREDIT_LIMIT' in cluster_means and cluster_means['CREDIT_LIMIT'] > 0:
            credit_utilization = (cluster_means['BALANCE'] / cluster_means['CREDIT_LIMIT']) * 100
            rec['characteristics'].append(f"Average Credit Utilization: {credit_utilization:.2f}%")
            if credit_utilization > 70:
                rec['cluster_name'] += " - High Credit Users"
                rec['characteristics'].append("High credit utilization.")
                rec['actions'].append("Monitor for credit risk, but also potential for balance transfer offers if payments are consistent.")
            elif credit_utilization < 20 and cluster_means['CREDIT_LIMIT'] > 1000 :
                 rec['actions'].append("Encourage responsible credit limit usage for benefits.")


        # Example: Frequent Purchases
        if 'PURCHASES_FREQUENCY' in cluster_means and cluster_means['PURCHASES_FREQUENCY'] > 0.75:
            rec['cluster_name'] += " - Frequent Shoppers"
            rec['characteristics'].append(f"High Purchase Frequency (avg: {cluster_means['PURCHASES_FREQUENCY']:.2f})")
            rec['actions'].append("Engage with regular offers and updates on new products.")

        # Example: High Cash Advance Users
        if 'CASH_ADVANCE' in cluster_means and cluster_means['CASH_ADVANCE'] > 500: # Example threshold
            rec['cluster_name'] += " - Cash Advance Users"
            rec['characteristics'].append(f"Significant Cash Advance usage (avg: ${cluster_means['CASH_ADVANCE']:.2f})")
            rec['actions'].append("Assess reasons for cash advance; offer alternative credit products if appropriate.")
            rec['actions'].append("Ensure awareness of cash advance fees and interest rates.")

        # Example: Low Activity / Dormant
        if 'PURCHASES_TRX' in cluster_means and cluster_means['PURCHASES_TRX'] < 5 and \
           'BALANCE' in cluster_means and cluster_means['BALANCE'] < 500: # Example thresholds
            rec['cluster_name'] += " - Low Activity"
            rec['characteristics'].append("Low transaction activity.")
            rec['characteristics'].append(f"Low Average Balance (${cluster_means['BALANCE']:.2f})")
            rec['actions'].append("Launch re-engagement campaigns with special offers or incentives.")

        # Add a default characteristic/action if none matched significantly
        if not rec['characteristics']:
            rec['characteristics'].append("This cluster shows a mixed or average profile across primary features.")
        if not rec['actions']:
            rec['actions'].append("Further analysis of specific feature interactions may reveal more targeted strategies.")
            rec['actions'].append("Consider general customer satisfaction surveys for this group.")

        recommendations.append(rec)

    return recommendations


# --- Helper Functions for Upload Route ---
def _validate_uploaded_file(file_storage):
    """
    Validates the uploaded file.
    Saves it temporarily if basic checks pass.
    Returns (is_valid, file_path_or_none, error_json_or_none)
    """
    if not file_storage: # Handles 'file' not in request.files case if called like _validate_uploaded_file(request.files.get('file'))
        return False, None, jsonify({'success': False, 'error': 'No file part in the request. Please select a file.'})

    if file_storage.filename == '':
        return False, None, jsonify({'success': False, 'error': 'No file selected. Please choose a CSV file to upload.'})

    if not file_storage.filename.lower().endswith('.csv'):
        return False, None, jsonify({'success': False, 'error': 'Invalid file type. Only .csv files are accepted.'})

    unique_filename_str = str(uuid.uuid4()) + "_" + file_storage.filename
    uploaded_file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename_str

    try:
        file_storage.save(uploaded_file_path)
        logging.info(f"File temporarily saved for validation: {uploaded_file_path}")

        if uploaded_file_path.stat().st_size == 0:
            os.remove(uploaded_file_path)
            logging.warning(f"Uploaded file {uploaded_file_path} is empty.")
            return False, None, jsonify({'success': False, 'error': 'The uploaded CSV file is empty. Please upload a file with data.'})

        try:
            temp_df = pd.read_csv(uploaded_file_path, nrows=5)
            if temp_df.empty and uploaded_file_path.stat().st_size > 0: # Check if it's not just headers
                 # If nrows=0, pandas might read it as empty. If size > 0 but df is empty, it's likely just headers or problematic.
                 # For a more robust check, one might try reading without nrows limit if nrows=5 yields empty.
                 # However, if nrows=5 gives an empty df for a non-zero size file, it's suspicious.
                 pass # Allow files that might only have headers and few rows for now, pipeline should handle
            # Consider a case where a file has content but it's not valid CSV (e.g. a binary file renamed to .csv)
            # pandas.read_csv will likely raise an error for that, caught by the outer except.
        except pd.errors.EmptyDataError: # This means no columns to parse, truly empty or malformed
            os.remove(uploaded_file_path)
            logging.warning(f"Uploaded file {uploaded_file_path} is empty or unparsable (EmptyDataError).")
            return False, None, jsonify({'success': False, 'error': 'The uploaded CSV file appears to be empty or corrupted (no columns to parse).'})
        except Exception as pe: # Catch other pandas parsing errors
            os.remove(uploaded_file_path)
            logging.warning(f"Uploaded file {uploaded_file_path} could not be parsed: {pe}")
            return False, None, jsonify({'success': False, 'error': f'Error parsing CSV file: {str(pe)}. Please check the file format.'})

        return True, uploaded_file_path, None # Validation successful

    except Exception as e_save:
        logging.error(f"Error saving uploaded file for validation: {e_save}")
        if uploaded_file_path.exists(): # Clean up if save started but failed
            try:
                os.remove(uploaded_file_path)
            except Exception: pass
        return False, None, jsonify({'success': False, 'error': f'Server error during file upload: {str(e_save)}'})

def _run_batch_prediction(input_csv_path: Path):
    """Runs the batch prediction pipeline."""
    prediction_output_csv_path = Path(app.config['PREDICTION_FOLDER']) / BATCH_PREDICTION_OUTPUT_FILE
    batch_config = BatchPredictionConfig(
        input_csv_path=input_csv_path,
        output_csv_path=prediction_output_csv_path,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH
    )
    logging.info("Initiating Batch Prediction Pipeline...")
    batch_pipeline = BatchPredictionPipeline(config=batch_config)
    return batch_pipeline.initiate_batch_prediction()

def _prepare_prediction_response_data(results_df: pd.DataFrame,
                                      original_input_file_path: Path,
                                      original_filename: str):
    """Prepares all data needed for the JSON success response."""

    summary_insights = generate_summary_insights(results_df.copy())

    # Prepare data for Chart.js
    cluster_dist_data = {}
    if 'cluster_distribution' in summary_insights:
        cluster_dist_data = {
            'labels': [f"Cluster {k}" for k in sorted(summary_insights['cluster_distribution'].keys())],
            'data': [summary_insights['cluster_distribution'][k] for k in sorted(summary_insights['cluster_distribution'].keys())]
        }

    scatter_plot_data = {'datasets': []}
    scatter_feature_x = 'Feature_1_Scaled'
    scatter_feature_y = 'Feature_2_Scaled'
    numeric_cols = results_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'CLUSTER']

    if len(numeric_cols) >= 2:
        scatter_feature_x = numeric_cols[0]
        scatter_feature_y = numeric_cols[1]
        for cluster_id in sorted(results_df['CLUSTER'].unique()):
            cluster_df = results_df[results_df['CLUSTER'] == cluster_id]
            scatter_plot_data['datasets'].append({
                'label': f'Cluster {cluster_id}',
                'data': list(zip(cluster_df[scatter_feature_x], cluster_df[scatter_feature_y]))
            })

    radar_chart_data = {'labels': [], 'datasets': []}
    radar_features = [col for col in numeric_cols if col not in [scatter_feature_x, scatter_feature_y]][:5]
    if not radar_features and len(numeric_cols) > 0:
        radar_features = numeric_cols[:min(5, len(numeric_cols))]

    if radar_features:
        radar_chart_data['labels'] = radar_features
        cluster_means = results_df.groupby('CLUSTER')[radar_features].mean()
        for cluster_id in sorted(results_df['CLUSTER'].unique()):
            if cluster_id in cluster_means.index:
                radar_chart_data['datasets'].append({
                    'label': f'Cluster {cluster_id}',
                    'data': cluster_means.loc[cluster_id].tolist()
                })

    visuals_data_for_json = {
        'cluster_distribution': cluster_dist_data,
        'scatter_plot': {'data': scatter_plot_data, 'feature_x': scatter_feature_x, 'feature_y': scatter_feature_y} if scatter_plot_data['datasets'] else {},
        'radar_chart': radar_chart_data if radar_chart_data['datasets'] else {}
    }

    results_table_for_html = results_df.to_html(classes='table table-striped table-hover table-sm', escape=False, max_rows=20, index=False)
    results_section_html = render_partial_to_string('_results_section.html',
                                                    results_table=results_table_for_html,
                                                    uploaded_filename=original_filename,
                                                    prediction_filename=BATCH_PREDICTION_OUTPUT_FILE)
    insights_section_html = render_partial_to_string('_insights_section.html', insights=summary_insights)
    visualizations_section_html = render_partial_to_string('_visualizations_section.html', visuals_data_json=visuals_data_for_json)

    recommendations_data = []
    try:
        original_df_for_insights = pd.read_csv(original_input_file_path)
        if len(original_df_for_insights) == len(results_df):
            original_df_for_insights['CLUSTER'] = results_df['CLUSTER']
            from custsegments.constant.training_pipeline import NUMERICAL_COLUMNS as SCHEMA_NUMERICAL_COLS
            recommendations_data = generate_business_recommendations(original_df_for_insights, SCHEMA_NUMERICAL_COLS)
        else:
            logging.warning("Could not merge cluster labels with original data for insights due to length mismatch.")
    except Exception as e_insights:
        logging.error(f"Error generating business insights: {e_insights}")

    recommendations_section_html = render_partial_to_string('_recommendations_section.html', recommendations=recommendations_data)

    return {
        'success': True,
        'message': 'Prediction successful!',
        'uploaded_filename': original_filename,
        'prediction_filename': BATCH_PREDICTION_OUTPUT_FILE,
        'results_section_html': results_section_html,
        'insights_section_html': insights_section_html,
        'visualizations_section_html': visualizations_section_html,
        'recommendations_section_html': recommendations_section_html,
        'visuals_data_json': visuals_data_for_json
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, triggers batch prediction, and returns JSON response."""

    file_storage = request.files.get('file')
    is_valid, temp_uploaded_csv_path, error_response = _validate_uploaded_file(file_storage)

    if not is_valid:
        # _validate_uploaded_file already returns a jsonify-ed response
        return error_response, 400

    try:
        prediction_artifact = _run_batch_prediction(temp_uploaded_csv_path)

        if prediction_artifact.is_prediction_successful and prediction_artifact.output_file_path:
            logging.info("Batch prediction successful.")
            results_df = read_csv(prediction_artifact.output_file_path)
            
            response_data = _prepare_prediction_response_data(results_df,
                                                              temp_uploaded_csv_path, # Pass path for re-reading original data
                                                              file_storage.filename) # Pass original filename
            return jsonify(response_data)
        else:
            logging.error(f"Prediction failed: {prediction_artifact.message}")
            return jsonify({'success': False, 'error': f'Error during prediction: {prediction_artifact.message}'}), 500

    except FileNotFoundError as fnf_error: # Model/preprocessor not found during BatchPredictionConfig
        logging.error(f"File not found error during prediction setup: {fnf_error}")
        return jsonify({'success': False, 'error': f"Error: A required file/model was not found. Ensure model and preprocessor are correctly set up. Details: {str(fnf_error)}"}), 500
    except CustomException as ce:
        logging.error(f"CustomException during upload/prediction: {ce}")
        return jsonify({'success': False, 'error': f"An application error occurred: {str(ce)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error during upload/prediction: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': f"An unexpected server error occurred: {str(e)}"}), 500
    finally:
        # Clean up the originally uploaded file if it exists
        if temp_uploaded_csv_path and temp_uploaded_csv_path.exists(): # Check if path was defined and exists
             try:
                 os.remove(temp_uploaded_csv_path)
                 logging.info(f"Cleaned up temporary uploaded file: {temp_uploaded_csv_path}")
             except Exception as e_clean:
                 logging.error(f"Error cleaning up temporary file {temp_uploaded_csv_path}: {e_clean}")

@app.route('/download_prediction/<filename>')
def download_prediction(filename: str):
    """Allows downloading the prediction output file."""
    try:
        # PREDICTION_FOLDER is customer_segments/prediction_output/
        prediction_file_path = Path(app.config['PREDICTION_FOLDER'])
        logging.info(f"Attempting to send file: {filename} from directory: {prediction_file_path}")
        return send_from_directory(directory=prediction_file_path, path=filename, as_attachment=True)
    except FileNotFoundError:
        flash("Predicted file not found.")
        logging.error(f"File not found for download: {filename} in {prediction_file_path}")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error downloading prediction file: {e}")
        flash("Error downloading file.")
        return redirect(url_for('index'))


# --- main.py functionality (running training pipeline) ---
# This could be a separate CLI command, but for simplicity, can be triggered via a route if needed for demo
@app.route('/run_training', methods=['GET']) # GET for simplicity, POST might be better
def run_training_pipeline_route():
    """Triggers the training pipeline."""
    try:
        flash("Training pipeline initiated... This may take some time. Check logs for progress.")
        logging.info("--- Flask: Training Pipeline Triggered ---")
        
        # For a real app, run this in a background thread/task queue (e.g., Celery)
        # to avoid blocking the web server.
        # For this demo, running it synchronously.
        
        # Ensure data is available for training pipeline
        original_data_path = Path(PROJECT_ROOT_DIR) / "customer_data" / "credit_card_data.csv"
        CUSTOMER_DATA_PARENT_DIR = Path(PROJECT_ROOT_DIR) / "customer_data"
        target_customer_data_path = Path(CUSTOMER_DATA_PARENT_DIR) / "credit_card_data.csv"
        if not target_customer_data_path.exists() and original_data_path.exists():
            os.makedirs(target_customer_data_path.parent, exist_ok=True)
            import shutil
            shutil.copy(original_data_path, target_customer_data_path)
            logging.info(f"Copied data for training: {target_customer_data_path}")
        elif not target_customer_data_path.exists():
             flash("Error: Training data not found. Cannot start pipeline.")
             logging.error("Training data not found for pipeline execution.")
             return redirect(url_for('index'))


        from datetime import datetime
        from custsegments.pipeline.training_pipeline import TrainingPipeline
        pipeline_run_ts = datetime.now().strftime("%Y%m%d%H%M%S") # Generate a new timestamp for this run
        training_pipeline = TrainingPipeline(run_timestamp=pipeline_run_ts)
        training_pipeline.run_pipeline() # This will log its own success/failure
        
        flash(f"Training pipeline run ({pipeline_run_ts}) completed. Check logs for details. Model and preprocessor may have been updated.")
        logging.info(f"--- Flask: Training Pipeline Run {pipeline_run_ts} Finished ---")
    except Exception as e:
        flash(f"Error running training pipeline: {str(e)}")
        logging.error(f"Error running training pipeline from Flask: {e}\n{traceback.format_exc()}")
    
    return redirect(url_for('index'))


if __name__ == "__main__":
    # Ensure critical model/preprocessor files exist for the app to start meaningfully
    # This is a basic check; a more robust app would handle their absence gracefully or guide setup.
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        warning_msg = (
            f"Warning: Model ({MODEL_PATH}) or Preprocessor ({PREPROCESSOR_PATH}) not found. "
            "Batch prediction will fail. Please run the training pipeline first (e.g., via /run_training endpoint or main.py)."
        )
        print(warning_msg, file=sys.stderr) # Print to stderr for visibility if running directly
        logging.warning(warning_msg)
        # For a demo, we might allow the app to start anyway.
        # flash(warning_msg) # Flash won't work before first request

    app.run(host="0.0.0.0", port=8080, debug=True) # debug=True for development

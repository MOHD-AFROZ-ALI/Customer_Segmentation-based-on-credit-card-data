import os
import sys
import uuid
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import pandas as pd
import traceback

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
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        try:
            # Save uploaded file with a unique name to avoid conflicts
            unique_filename = str(uuid.uuid4()) + "_" + file.filename
            uploaded_csv_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
            file.save(uploaded_csv_path)
            logging.info(f"File uploaded successfully: {uploaded_csv_path}")

            # Define output path for this specific prediction run
            # The filename in PREDICTION_FOLDER could also be made unique if multiple users access concurrently
            # For simplicity, let's use the constant name but inside a timestamped or UUID folder if needed.
            # For now, just use the standard output name.
            prediction_output_csv_path = Path(app.config['PREDICTION_FOLDER']) / BATCH_PREDICTION_OUTPUT_FILE

            # Configure Batch Prediction
            batch_config = BatchPredictionConfig(
                input_csv_path=uploaded_csv_path,
                output_csv_path=prediction_output_csv_path,
                model_path=MODEL_PATH,
                preprocessor_path=PREPROCESSOR_PATH
            )
            
            logging.info("Initiating Batch Prediction Pipeline...")
            batch_pipeline = BatchPredictionPipeline(config=batch_config)
            prediction_artifact = batch_pipeline.initiate_batch_prediction()

            if prediction_artifact.is_prediction_successful and prediction_artifact.output_file_path:
                logging.info("Batch prediction successful.")
                # Read the clustered results to display
                results_df = read_csv(prediction_artifact.output_file_path)
                
                # Generate summary insights
                summary_insights = generate_summary_insights(results_df.copy()) # Use a copy for safety
                
                # Placeholder for visuals - e.g., a bar chart of cluster distribution
                # This would typically involve generating an image and passing its path or embedding it.
                # For simplicity, we'll just pass the data for now.
                visuals_data = summary_insights.get('cluster_distribution', {})

                # Clean up uploaded file after processing
                # os.remove(uploaded_csv_path)
                # logging.info(f"Removed uploaded file: {uploaded_csv_path}")
                
                # Store results in session or pass to template directly
                # For large results, consider pagination or saving to a temporary DB
                return render_template('index.html', 
                                       results=results_df.to_html(classes='table table-striped', escape=False, max_rows=20),
                                       insights=summary_insights,
                                       visuals_data=visuals_data, # Pass data for JS charting
                                       uploaded_filename=file.filename,
                                       prediction_filename=BATCH_PREDICTION_OUTPUT_FILE)
            else:
                flash(f'Error during prediction: {prediction_artifact.message}')
                logging.error(f"Prediction failed: {prediction_artifact.message}")
                return redirect(url_for('index'))

        except FileNotFoundError as fnf_error:
            logging.error(f"File not found error: {fnf_error}")
            flash(f"Error: A required file was not found. Please ensure model and preprocessor are in '{FINAL_MODEL_DIR}'. Details: {str(fnf_error)}")
            return redirect(url_for('index'))
        except CustomException as ce:
            logging.error(f"CustomException during upload/prediction: {ce}")
            flash(f"An error occurred: {str(ce)}")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Unexpected error during upload/prediction: {e}\n{traceback.format_exc()}")
            flash(f"An unexpected error occurred: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a CSV file.')
        return redirect(request.url)

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

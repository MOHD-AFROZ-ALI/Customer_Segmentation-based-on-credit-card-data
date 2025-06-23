import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback


from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
from custsegments.entity.artifact_entity import DataTransformationArtifact # EDA typically uses transformed data
# Config for EDA might be needed if we have specific parameters, for now, paths are derived or passed
from custsegments.utils.main_utils.utils import read_csv, create_directories,save_object
from custsegments.constant.training_pipeline import EDA_PLOTS_DIR, EDA_REPORT_FILE # Constants for output paths


class EDAAnalysis:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 eda_output_plots_dir: Path, # Specific directory for plots for this run
                 eda_report_file_path: Path): # Specific file for report for this run
        self.transformation_artifact = data_transformation_artifact
        self.plots_dir = eda_output_plots_dir
        self.report_file = eda_report_file_path
        create_directories([self.plots_dir, self.report_file.parent])

    def generate_descriptive_statistics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates descriptive statistics for the dataframe."""
        try:
            logging.info("Generating descriptive statistics...")
            desc_stats = dataframe.describe(include='all').transpose()
            logging.info(f"Descriptive statistics generated:\n{desc_stats}")
            return desc_stats
        except Exception as e:
            logging.error(f"Error generating descriptive statistics: {e}")
            raise CustomException(e, sys) from e

    def generate_visualizations(self, dataframe: pd.DataFrame):
        """Generates and saves various visualizations for EDA."""
        try:
            logging.info("Generating visualizations...")
            
            # Histograms for numerical features
            numerical_cols = dataframe.select_dtypes(include=np.number).columns
            for col in numerical_cols:
                plt.figure(figsize=(8, 5))
                sns.histplot(dataframe[col], kde=True)
                plt.title(f'Histogram of {col}')
                plot_path = self.plots_dir / f'{col}_histogram.png'
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved histogram for {col} to {plot_path}")

            # Box plots for numerical features
            for col in numerical_cols:
                plt.figure(figsize=(8, 5))
                sns.boxplot(y=dataframe[col])
                plt.title(f'Box Plot of {col}')
                plot_path = self.plots_dir / f'{col}_boxplot.png'
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved box plot for {col} to {plot_path}")

            # Correlation matrix heatmap (only for numerical columns)
            if len(numerical_cols) > 1:
                plt.figure(figsize=(12, 10))
                correlation_matrix = dataframe[numerical_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title('Correlation Matrix of Numerical Features')
                plot_path = self.plots_dir / 'correlation_matrix.png'
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved correlation matrix heatmap to {plot_path}")
            
            # Pairplots for a subset of features (can be resource-intensive for many features)
            # Example: Select a few key features for pairplot or make it configurable
            if len(numerical_cols) > 1 and len(numerical_cols) <= 5: # Limit for practical reasons
                logging.info(f"Generating pairplot for features: {list(numerical_cols)}")
                pairplot_fig = sns.pairplot(dataframe[numerical_cols].sample(min(1000, len(dataframe))), diag_kind='kde') # Sample for large datasets
                plot_path = self.plots_dir / 'pairplot.png'
                pairplot_fig.savefig(plot_path)
                plt.close()
                logging.info(f"Saved pairplot to {plot_path}")
            elif len(numerical_cols) > 5:
                logging.warning("Skipping pairplot due to large number of numerical features (>5). Consider selecting a subset.")

            

            logging.info("Visualizations generated and saved.")
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
            raise CustomException(e, sys) from e
            
    def generate_eda_report(self, dataframe: pd.DataFrame, desc_stats: pd.DataFrame):
        """
        Generates a textual EDA report summarizing findings.
        """
        try:
            logging.info("Generating EDA report...")
            with open(self.report_file, 'w') as f:
                f.write("Exploratory Data Analysis Report\n")
                f.write("=================================\n\n")
                
                f.write("1. Dataset Overview:\n")
                f.write(f"   - Number of records: {dataframe.shape[0]}\n")
                f.write(f"   - Number of features: {dataframe.shape[1]}\n\n")
                
                f.write("2. Descriptive Statistics:\n")
                f.write(desc_stats.to_string())
                f.write("\n\n")
                
                f.write("3. Null Value Analysis (from transformed data - should be 0 after imputation):\n")
                null_summary = dataframe.isnull().sum()
                null_summary = null_summary[null_summary > 0]
                if not null_summary.empty:
                    f.write("   Null values found in the (supposedly transformed) data:\n")
                    f.write(null_summary.to_string())
                    f.write("\n\n")
                else:
                    f.write("   No null values found in the transformed data (as expected after imputation).\n\n")

                f.write("4. Initial Insights & Patterns (Qualitative):\n")
                f.write("   - [Placeholder] Observe distributions from histograms (skewness, modality).\n")
                f.write("   - [Placeholder] Identify potential outliers from box plots.\n")
                f.write("   - [Placeholder] Note strong correlations (positive/negative) from the heatmap.\n")
                f.write("   - [Placeholder] Visual patterns from pairplots (if generated) suggesting clusters or relationships.\n")
                f.write("   - Note: This EDA is on *transformed* (scaled, imputed) data. Original scale insights might differ.\n")

            logging.info(f"EDA report saved to {self.report_file}")
        except Exception as e:
            logging.error(f"Error generating EDA report: {e}")
            raise CustomException(e, sys) from e

    def initiate_eda_analysis(self):
        """
        Main method to orchestrate the EDA process.
        """
        logging.info("Starting EDA analysis process.")
        try:
            if not self.transformation_artifact.is_transformed or not self.transformation_artifact.transformed_data_path:
                message = "Data transformation was not successful or transformed data path is missing. Skipping EDA."
                logging.warning(message)
                # Optionally, write a minimal report indicating EDA was skipped
                with open(self.report_file, 'w') as f:
                    f.write(f"EDA Analysis Report\n===================\n{message}\n")
                return # Or raise an exception if EDA is critical

            # Load transformed data for EDA
            # EDA is often performed on data *before* extensive transformation (like scaling) to understand original distributions,
            # but for this pipeline, it's placed after transformation. We'll use the transformed data.
            # If EDA on original data is needed, the component would need access to an earlier artifact.
            df = read_csv(self.transformation_artifact.transformed_data_path)
            logging.info(f"Loaded transformed data for EDA from: {self.transformation_artifact.transformed_data_path}")

            desc_stats = self.generate_descriptive_statistics(df)
            self.generate_visualizations(df)
            self.generate_eda_report(df, desc_stats)
            
            logging.info("EDA analysis completed successfully.")
            # EDA component typically doesn't produce a structured artifact like other components,
            # its outputs are reports and plots. We can consider creating an EDArtifact if specific paths need to be passed on.

        except Exception as e:
            logging.error(f"Error during EDA analysis: {e}")
            raise CustomException(e, sys) from e


if __name__ == '__main__':
    from custsegments.constant.training_pipeline import (
        ARTIFACTS_DIR, PIPELINE_NAME, DATA_TRANSFORMATION_DIR_NAME, 
        TRANSFORMED_DATA_FILE, PREPROCESSOR_OBJECT_FILE_NAME,
        EDA_ANALYSIS_ROOT_DIR, EDA_PLOTS_DIR as EDA_PLOTS_DIR_CONST, # Renamed to avoid conflict
        EDA_REPORT_FILE as EDA_REPORT_FILE_CONST
    )
    import shutil
    
    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent.parent 

    # Use a unique artifacts root for this test to avoid conflicts
    test_artifacts_root = project_root_for_test / ARTIFACTS_DIR / f"{PIPELINE_NAME}_test_eda"

    # --- Dummy DataTransformationArtifact ---
    dummy_transformed_data_path = test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME / TRANSFORMED_DATA_FILE
    dummy_preprocessor_path = test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME / PREPROCESSOR_OBJECT_FILE_NAME
    
    os.makedirs(dummy_transformed_data_path.parent, exist_ok=True)
    # Create a more comprehensive dummy transformed CSV for EDA
    data = {
        'BALANCE': np.random.rand(100) * 10000,
        'PURCHASES': np.random.rand(100) * 5000,
        'CREDIT_LIMIT': np.random.rand(100) * 15000,
        'PAYMENTS': np.random.rand(100) * 20000,
        'TENURE': np.random.randint(6, 13, 100)
    }
    # Simulate scaled data (mean ~0, std ~1)
    for col in ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']:
        data[col] = (data[col] - np.mean(data[col])) / np.std(data[col])

    dummy_df_transformed = pd.DataFrame(data)
    dummy_df_transformed.to_csv(dummy_transformed_data_path, index=False)
    logging.info(f"Created dummy transformed data at {dummy_transformed_data_path} for EDAAnalysis test.")
    
    # Create dummy preprocessor file (content doesn't matter for this test)
    save_object(dummy_preprocessor_path, {"dummy": "preprocessor"})

    data_transformation_artifact_test = DataTransformationArtifact(
        transformed_data_path=dummy_transformed_data_path,
        preprocessor_object_path=dummy_preprocessor_path,
        is_transformed=True,
        message="Dummy data transformation artifact for EDA test."
    )

    # --- Configuration for EDAAnalysis ---
    # EDA_ANALYSIS_ROOT_DIR is ARTIFACTS_DIR / PIPELINE_NAME / EDA_ANALYSIS_DIR_NAME
    # EDA_PLOTS_DIR_CONST is EDA_ANALYSIS_ROOT_DIR / "plots"
    # EDA_REPORT_FILE_CONST is EDA_ANALYSIS_ROOT_DIR / "eda_report.txt"

    eda_run_plots_dir = test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME / "eda_plots" # Store plots within this test's artifact structure
    eda_run_report_file = test_artifacts_root / DATA_TRANSFORMATION_DIR_NAME / "eda_test_report.txt"


    logging.info(f"EDA Plots Dir for test: {eda_run_plots_dir}")
    logging.info(f"EDA Report File for test: {eda_run_report_file}")

    try:
        eda_analysis = EDAAnalysis(
            data_transformation_artifact=data_transformation_artifact_test,
            eda_output_plots_dir=eda_run_plots_dir,
            eda_report_file_path=eda_run_report_file
        )
        eda_analysis.initiate_eda_analysis()
        logging.info("EDAAnalysis component test finished successfully.")

        assert eda_run_plots_dir.exists(), "EDA plots directory was not created."
        assert eda_run_report_file.exists(), "EDA report file was not created."
        # Check if some plot files were created
        assert len(list(eda_run_plots_dir.glob('*.png'))) > 0, "No PNG plot files found in EDA plots directory."
        
        logging.info(f"Plots saved in: {eda_run_plots_dir}")
        logging.info(f"Report saved at: {eda_run_report_file}")
        
        # Clean up (optional)
        # if test_artifacts_root.exists():
        #    shutil.rmtree(test_artifacts_root)
        #    logging.info(f"Cleaned up test artifacts directory: {test_artifacts_root}")

    except CustomException as ce:
        logging.error(f"EDAAnalysis component test failed: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error in EDAAnalysis component test: {e}")
        logging.error(traceback.format_exc())

import os
import yaml
import json
import pickle
import dill # Using dill for wider object serialization support, if needed. pickle is often fine.
from pathlib import Path
from typing import Any, Union # For type hinting
import pandas as pd
import traceback

from custsegments.logging.logger import logging
from custsegments.exception.exception import CustomException
import sys


def read_yaml(path_to_yaml: Path) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        dict: Content of the YAML file.
    
    Raises:
        CustomException: If there's an error reading the file.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
        logging.info(f"YAML file: {path_to_yaml} loaded successfully")
        return content
    except Exception as e:
        raise CustomException(e, sys) from e

def write_yaml(file_path: Path, data: dict, overwrite: bool = False):
    """
    Writes data to a YAML file.

    Args:
        file_path (Path): Path to the YAML file.
        data (dict): Data to write.
        overwrite (bool, optional): Whether to overwrite if the file exists. Defaults to False.
    
    Raises:
        CustomException: If there's an error writing the file.
    """
    try:
        if overwrite or not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
            logging.info(f"YAML file saved at: {file_path}")
        else:
            logging.info(f"File {file_path} already exists. Set overwrite=True to replace.")
    except Exception as e:
        raise CustomException(e, sys) from e

def save_object(file_path: Path, obj: Any):
    """
    Saves a Python object to a file using dill (or pickle).

    Args:
        file_path (Path): Path to save the object.
        obj (Any): The Python object to save.
    
    Raises:
        CustomException: If there's an error saving the object.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path: Path) -> Any:
    """
    Loads a Python object from a file using dill (or pickle).

    Args:
        file_path (Path): Path to the object file.

    Returns:
        Any: The loaded Python object.

    Raises:
        CustomException: If there's an error loading the object.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found at {file_path}")
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e

def save_json(path: Path, data: dict):
    """
    Saves data to a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save.
    
    Raises:
        CustomException: If there's an error saving the JSON.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e
        
def load_json(path: Path) -> dict:
    """
    Loads data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
        
    Raises:
        CustomException: If there's an error loading the JSON.
    """
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        logging.info(f"JSON file loaded successfully from: {path}")
        return content
    except Exception as e:
        raise CustomException(e, sys) from e

def create_directories(path_to_directories: list[Union[str, Path]], verbose=True):
    """
    Creates a list of directories.

    Args:
        path_to_directories (list): List of directory paths.
        verbose (bool, optional): Whether to log creation. Defaults to True.
    """
    for path in path_to_directories:
        path = Path(path) # Ensure it's a Path object
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")

def get_size(path: Path) -> str:
    """
    Gets the size of a file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def read_csv(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): Path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pd.read_csv.

    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        CustomException: If there's an error reading the CSV.
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        logging.info(f"CSV file loaded successfully from: {file_path}")
        return df
    except Exception as e:
        raise CustomException(e, sys) from e

def save_csv(df: pd.DataFrame, file_path: Path, index: bool = False, **kwargs):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (Path): Path to save the CSV file.
        index (bool, optional): Whether to write DataFrame index. Defaults to False.
        **kwargs: Additional keyword arguments to pass to df.to_csv.
        
    Raises:
        CustomException: If there's an error saving the CSV.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=index, **kwargs)
        logging.info(f"DataFrame saved to CSV at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e

if __name__ == '__main__':
    # Example usage (optional, for testing)
    try:
        # Create dummy directories and files for testing
        test_log_dir = Path("C:\\Users\\hp\\Desktop\\CUSTO\\customer_segments_productionjules\\custsegments\\utils\\main_utils\\test_logs") # Relative to where it might be run from
        test_artifacts_dir = Path("C:\\Users\\hp\\Desktop\\CUSTO\\customer_segments_productionjules\\custsegments\\utils\\main_utils\\test_artifacts")

        create_directories([test_log_dir, test_artifacts_dir])

        # Test YAML
        yaml_data = {'name': 'test', 'version': 1}
        yaml_file_path = test_artifacts_dir / "test.yaml"
        write_yaml(yaml_file_path, yaml_data, overwrite=True)
        loaded_yaml_data = read_yaml(yaml_file_path)
        assert yaml_data == loaded_yaml_data
        logging.info(f"YAML test successful: {loaded_yaml_data}")

        # Test JSON
        json_data = {'key': 'value', 'numbers': [1, 2, 3]}
        json_file_path = test_artifacts_dir / "test.json"
        save_json(json_file_path, json_data)
        loaded_json_data = load_json(json_file_path)
        assert json_data == loaded_json_data
        logging.info(f"JSON test successful: {loaded_json_data}")

        # Test Object save/load
        obj_data = {"a": 10, "b": [1,2,3], "c": "test_obj"}
        obj_file_path = test_artifacts_dir / "test.pkl"
        save_object(obj_file_path, obj_data)
        loaded_obj_data = load_object(obj_file_path)
        assert obj_data == loaded_obj_data
        logging.info(f"Object save/load test successful: {loaded_obj_data}")
        
        # Test CSV read/write
        data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data)
        csv_file_path = test_artifacts_dir / "test.csv"
        save_csv(df, csv_file_path)
        loaded_df = read_csv(csv_file_path)
        pd.testing.assert_frame_equal(df, loaded_df)
        logging.info(f"CSV read/write test successful:\n{loaded_df.head()}")

        logging.info("All main_utils tests passed successfully!")

    except CustomException as ce:
        logging.error(f"A custom exception occurred during testing: {ce}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during testing: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Clean up test directories and files (optional)
        # import shutil
        # if test_log_dir.exists(): shutil.rmtree(test_log_dir)
        # if test_artifacts_dir.exists(): shutil.rmtree(test_artifacts_dir)
        pass

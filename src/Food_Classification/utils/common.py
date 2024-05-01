import os
from box.exceptions import BoxValueError
import yaml
from Food_Classification import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """ Reads the yaml file and returns the ConfigBox object

    Args:
        file_path (Path): path to the yaml file

    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'yaml file {file_path} loaded successfully')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directory(directory_path: list,verbose: bool = True):
    """ Creates the directory if it does not exist.
    Args:
    directory_path (list): list of directories to be created.
    verbose (bool, optional): ignore if multiple direcotries to be created. Defaults to True.
    """

    for path in directory_path:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f'directory {path} created successfully')

@ensure_annotations
def load_json(file_path: Path) -> ConfigBox:
    """ Loads the json file and returns the data
    Args:
    file_path (Path): path to the json file

    returns:
    ConfigBox: ConfigBox object
    """
    with open(file_path) as json_file:
        content = json.load(json_file)
        logger.info(f'json file {file_path} loaded successfully')
        return ConfigBox(content)



@ensure_annotations

def save_json(file_path: Path, data: dict):
    """ Saves the data in the json file

    Args:
    file_path (Path): path to the json file
    data (dict): data to be saved in the json file
    """
    with open(file_path,'w') as json_file:
        json.dump(data, json_file, indent=4)

    logger.info(f'json file {file_path} saved successfully')


@ensure_annotations
def load_binary(file_path: Path):
    """ Loads the binary file and returns the data
    Args:
    file_path (Path): path to the binary file
    """
    data = joblib.load(file_path)
    logger.info(f'binary file {file_path} loaded successfully')
    return data

@ensure_annotations
def save_binary(file_path: Path, data: Any):
    """ Saves the data in the binary file

    Args:
    file_path (Path): path to the binary file
    data (Any): data to be saved in the binary file
    """
    joblib.dump(data, file_path)
    logger.info(f'binary file {file_path} saved successfully')


@ensure_annotations
def get_size(file_path: Path) -> str:
    """ get the size of the file in KB

    Args:
        file_path (Path): Path of the file

    Returns:
        str: size in KB
    """

    size_in_KB = round(os.path.getsize(file_path) / 1024)
    return f"~ {size_in_KB} KB"

    

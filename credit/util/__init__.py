# This module contains all the utility functions that might be used in other modules

import os,sys
from datetime import datetime
import yaml

from credit.exception import CreditException


def get_current_timestamp():
    """this function return current time stamp in %Y-%m-%d-%H-%M-%s format
    """

    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

def read_yaml_file(file_path:str)->dict:
    """This function reads YAML file and returns the contents as a dictionary"""
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditException(e,sys) from e


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise CreditException(e,sys) from e

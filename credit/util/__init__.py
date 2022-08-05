# This module contains all the utility functions that might be used in other modules

import os,sys
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import dill
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


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        #create directory path for file path
        dir_path = os.path.dirname(file_path)
        # checking if the dir exists if not create it
        os.makedirs(dir_path, exist_ok=True)
        #open filepath 
        with open(file_path, 'wb') as file_obj:
            #savin numpy array in file 
            np.save(file_obj, array)
    except Exception as e:
        raise CreditException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditException(e, sys) from e


def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CreditException(e,sys) from e

def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CreditException(e,sys) from e

def load_data(file_path:str, schema_file_path:str)-> pd.DataFrame:
    """This function reads csv file compare datatype of the columns with schema 
    and returns a DataFrame"""
    try:
        #reading schema file
        dataset_schema=read_yaml_file(schema_file_path)
        #loading columns dict in schema    
        schema = dataset_schema["columns"]
        #reading csv file
        dataframe = pd.read_csv(file_path)
        #Initailising error_message
        error_message = ""

        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message =f"{error_message} \nColumn:[{column}] is not in the schema."
        if len(error_message)>0:
            raise  Exception(error_message)
        return dataframe

    except Exception as e:
        raise CreditException(e,sys) from e    
import importlib

import numpy as np
import yaml
from credit.exception import CreditException
import os
import sys
from credit.constants import *
from collections import namedtuple
from typing import List
from credit.logger import logging
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,roc_auc_score

GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

MetricInfoArtifactClassifier = namedtuple("MetricInfoArtifactClassifier",
                                ["model_name", "model_object", "train_roc_auc", "test_roc_auc", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])




class ModelFactory:
    def __init__(self, model_config_path: str = None,):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object, property_data: dict):
        """This method takes parameter:
        instance_ref(object): class object
        property_data(dict): data of the attributes that needs to updated
        
        Returns 
        class object with updated parameters"""
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            logging.info(f"updating following data in the class : {property_data}")
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        """This method creates object of the class 
        parameters:
        module_name: eg. sklearn.LinearModel
        class_name: eg. LinearRegression
        """
        try:
            #load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise CreditException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        excute_grid_search_operation(): method will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        initialized_model(tuple): Model object with dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchBestModel object that has values
        "model_serial_number",
        "model",
        "best_model",
        "best_parameters",
        "best_score"


        """
        try:
            #Instantiating GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )
            #Initialising  GridSearchCV estimator and param_grid parameters
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            #updating paramaeters based on property data provided
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)

            
            message = f'\n\n{">>"* 30} "Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            #grid search fit implemented on input and output feature 
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>"* 30} Training {type(initialized_model.model).__name__} completed {"<<"*30}\n\n'
            #creating grid searched best model details 
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            #return the grid searched best model
            logging.info(f"grid searched best model : {type(grid_searched_best_model.model).__name__} with best parameters:{grid_searched_best_model.best_parameters}")
            return grid_searched_best_model
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This function will return a list of model details from model config file.
        return List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():

                model_initialization_config = self.models_initialization_config[model_serial_number]
                #creating model object by from module name and class name in model file 
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY]
                                                            )
                model = model_obj_ref()
                
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)
                #Creating paramter grid search dict from model config file
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                #Loading model name from model config file
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                #creating model initaialization config
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )
                #Appending initialised model list
                initialized_model_list.append(model_initialization_config)
            #updating instance attributes
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        Initiate_best_parameter_search_for initialised_model(): method will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        initialized_model(List): Model object with parameters to perform grid search
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:
        """ This method requires parameters:
        initialized_model_list(List): List of values in the form of InitializedModelDetail(tuple)
        input_feature(array): Transformed training dataset with input features
        output_feature(array): Transformed trainig dataset with output feature
         
         Returns
         grid searched best model (List)"""
        try:
            self.grid_searched_best_model_list = []
           #for every model finding the best parameters
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
           #returning all the models with best parameters
            return self.grid_searched_best_model_list
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This method take parameters:
        model_details:List[InitializedModelDetail]
        model_serial_number(str): model serial number 
        return InitializedModelDetail which has values

        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6
                                                          ) -> BestModel:
        """This method takes paramters:
        grid searched best model list(list): list of models with best parameters 
        base accuracy(float): accuracy  
        ==========================================================================
        Returns BestModel 
         "model_serial_number",
        "model",
        "best_model",
        "best_parameters",
        "best_score"
        """
        try:
            best_model = None
           #for every model in grid searched best model list if best score is greater than base accuracy it is selected
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                   #updating base accuracy
                    base_accuracy = grid_searched_best_model.best_score
                    #selecting best model
                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy greater than or equal to: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_best_model(self, X, y,base_accuracy=0.6) -> BestModel:
        """ This method require parameters:
        X: Input features array
        y: Target feature array
        base_accuracy(float):default 0.6
        Returns
        
        "model_serial_number",
        "model",
        "best_model",
        "best_parameters",
        "best_score" """
        try:
            #Getting models from model config file
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            #getting grid searched best model for each model type
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            #return the best model among the grid searched models
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise CreditException(e, sys)


def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6)->MetricInfoArtifactClassifier:
    """
    Description:
    This function compare classification models return best model
    Params:
    model_list: List of models
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    ===========================================================================
    return
    It retured a named tuple
    
    MetricInfoArtifactClassifier = namedtuple("MetricInfoArtifactClassifier",
                                ["model_name", "model_object", "train_roc_auc", "test_roc_auc", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
    """
    try:
        index_number = 0
        metric_info_artifact = None

        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"\n{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")

             #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #Calculating accuracy score on training and testing dataset
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            #Calculating roc_auc_acore on training and testing dataset
            train_roc_auc = roc_auc_score(y_train, y_train_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_pred)

            # Calculating harmonic mean of train_accuracy and test_accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            
            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
            logging.info(f"Train roc_auc_score: [{train_roc_auc}].")
            logging.info(f"Test roc_auc_score: [{test_roc_auc}].")

            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifactClassifier(model_name=model_name,
                                                        model_object=model,
                                                        train_roc_auc=train_roc_auc,
                                                        test_roc_auc=test_roc_auc,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)
                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact


    except Exception as e:
        raise CreditException(e,sys) from e


def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model
    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    return
    It retured a named tuple
    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
    """
    try:
        
    
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            
            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #Calculating r squared score on training and testing dataset
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)
            
            #Calculating mean squared error on training and testing dataset
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Calculating harmonic mean of train_accuracy and test_accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            
            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
            logging.info(f"Train root mean squared error: [{train_rmse}].")
            logging.info(f"Test root mean squared error: [{test_rmse}].")


            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise CreditException(e, sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    """This fuction creates sample model config yaml file in the export_dir
    and returns the path for that file"""
    try:
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }

            },
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY:
                        {"param_name1": "value1",
                         "param_name2": "value2",
                         },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }

                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise CreditException(e, sys)
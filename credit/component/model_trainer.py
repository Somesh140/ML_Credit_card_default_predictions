from typing import List

from sklearn import preprocessing
from credit.entity.model_factory import GridSearchedBestModel, MetricInfoArtifact, MetricInfoArtifactClassifier,\
     ModelFactory, evaluate_regression_model,evaluate_classification_model
from credit.exception import CreditException
import os,sys
from credit.logger import logging
from credit.entity.config_entity import DataTransformationConfig,ModelTrainerConfig
from credit.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact, ModelTrainerArtifactClassifier
import numpy as np
import pandas as pd
from credit.util import load_object,save_object,load_numpy_array_data
from credit.constants import *


class DefaultPredictorModel:

    def __init__(self,preprocessing_object,trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object= preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self,X):
        """
        function accepts raw inputs and then transform raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
                data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"\n\n{'>>'*30}Model Trainer log started.{'<<'*30}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise CreditException(e,sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifactClassifier:
        """ 1. In this method we are loading transformed training and testing dataset
            2. Reading the model config file 
            3. Getting best model on training dataset
            4. Evaluation models on both training and testing dataset-->model object
            5. Loading preprocessing object
            6. Creating custom model object combining both preprocessed object and model object 
            7. Saving the custom model object
            8. Returning model trainer artifact 
            Returns ModelTrainerArtifact(tuple) values 
            "is_trained"(bool):True|False, 
            "message"(str):, 
            "trained_model_file_path"(str): trained model file path,
            "train_roc_auc"(float): roc_auc_score on training dataset, 
            "test_roc_auc"(float): roc_auc_score on testing dataset, 
            "train_accuracy"(float):accuracy score on train dataset , 
            "test_accuracy"(float): accuracy score on test dataset,
            "model_accuracy"(float):Harmonic mean of train and test accuracy """
        
        try:
            #Loading transformed training dataset
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)
            #Loading transformed testing dataset
            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)
            #Splitting training and testing dataset into input and target feature
            logging.info(f"Splitting training and testing dataset into input and target feature")
            x_train,y_train,x_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            #creating model config file path
            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path
            #Initialising model factory class using  model config file
            logging.info(f"Initialising model factory class using above model config file:{model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            #loading base accuracy from config file
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy:{base_accuracy}")

            logging.info(f"Initiating operation model selection")
            best_model= model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)

            logging.info(f"Best model found on training dataset:{best_model}")

            logging.info(f"Extracting trained model list.")
            #getting Model Factory class attributes
            grid_searched_best_model:List[GridSearchedBestModel]=model_factory.grid_searched_best_model_list
            #creating model list of best models from grid searched models
            model_list =[model.best_model for model in grid_searched_best_model]

            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info:MetricInfoArtifactClassifier =evaluate_classification_model(
                                    model_list=model_list,
                                    X_train=x_train,
                                    y_train=y_train,
                                    X_test=x_test,
                                    y_test=y_test)

            logging.info(f"Best model found on  both training and testing dataset.{metric_info}")
            #loading data transformation preprocessing object
            preprocessing_obj=load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path) 
            #Loading training model object
            model_object = metric_info.model_object
            #creating trained model file path
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            #creating credit model predictor object 
            credit_model = DefaultPredictorModel(preprocessing_object=preprocessing_obj,
                                    trained_model_object=model_object)

            logging.info(f"Saving model at path:{trained_model_file_path}")
            #saving credit default predictor object 
            save_object(file_path=trained_model_file_path,obj=credit_model) 
            #creating model trainer artifact
            model_trainer_artifact_c = ModelTrainerArtifactClassifier(is_trained=True,
                                    message="Model Trained Successfully",
                                    trained_model_file_path = trained_model_file_path,
                                    train_roc_auc = metric_info.train_roc_auc,
                                    test_roc_auc=metric_info.test_roc_auc,
                                    train_accuracy = metric_info.train_accuracy,
                                    test_accuracy = metric_info.test_accuracy,
                                    model_accuracy = metric_info.model_accuracy)

            logging.info(f"Model Trainer Artifact:{model_trainer_artifact_c}")
            return model_trainer_artifact_c
        except Exception as e:
            raise CreditException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} \n")
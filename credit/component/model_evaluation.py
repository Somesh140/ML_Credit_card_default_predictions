from credit.config.configuration import configuration
from credit.entity.model_factory import MetricInfoArtifact, evaluate_regression_model,evaluate_classification_model
from credit.exception import CreditException
from credit.logger import logging
import os,sys
import numpy as np
from credit.util import load_data, load_object, read_yaml_file,save_object, write_yaml_file
from credit.entity.config_entity import ModelEvaluationConfig
from credit.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,\
     ModelEvaluationArtifact, ModelTrainerArtifact
from credit.constants import *


class ModelEvaluation:

    def __init__(self,model_evaluation_config:ModelEvaluationConfig,
                data_ingestion_artifact:DataIngestionArtifact,
                data_validation_artifact:DataValidationArtifact,
                model_trainer_artifact:ModelTrainerArtifact) :
        try:
            logging.info(f"\n\n{'>>'*20}Model Evaluation log started.{'<<'*30}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_best_model(self):
        """This method read the yaml file containing details for the best model trained
        returns best model instance"""
        try:
            model = None
           #create model evaluation file path
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            #create model evaluation dir and model evaluation yaml file it there exists none
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)

                return model
            #reading model evaluation yaml file
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            #creating empty dictionary in case yaml file is not there
            model_eval_file_content= dict() if model_eval_file_content is None else model_eval_file_content
            #searching best model in yaml file 
            if BEST_MODEL_KEY not in model_eval_file_content:
                return model
            #Instantiating best model object
            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise CreditException(e,sys) from e

    def update_evaluation_report(self,model_evaluation_artifact:ModelEvaluationArtifact):
        """This method updates model evaluation file with best model model location and history
        best model location"""
        try:
            #creating model evaluation file path
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            #reading model evaluation file content in to dictionary
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            #if there is no file creating empty dictionary
            model_eval_content = dict() if model_eval_content is None else model_eval_content
                       
            previous_best_model = None
            #finding best model in model evaluation file
            if BEST_MODEL_KEY in model_eval_content:
                #updating previous best model
                previous_best_model = model_eval_content[BEST_MODEL_KEY]
            
            logging.info(f"Previous evaluation result{model_eval_content}")
            #creating dictionary evaluation result
            eval_result = {
                BEST_MODEL_KEY :{
                    MODEL_PATH_KEY:model_evaluation_artifact.evaluated_model_path
                }
            }

            if previous_best_model is not None:
                #creatiing model history dictionary
                model_history ={self.model_evaluation_config.time_stamp:previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    #adding history of previous best model
                    history = {HISTORY_KEY:model_history}
                    #updating evaluation result dictionary
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
            #updating model evaluation dictionary
            model_eval_content.update(eval_result)
            logging.info(f"Update eval result:{model_eval_content}")
            #writing the content of model evaluation dictionary into model evaluation file
            write_yaml_file(file_path=eval_file_path,data=model_eval_content)
    
        except Exception as e:
            raise CreditException(e,sys) from e

    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        """This method initiates model evaluation 
        Returns 
        "is_model_accepted"(bool):True/False, 
        "evaluated_model_path(str)": model evaluation file path """
        try:
            trained_model_file_path =self.model_trainer_artifact.trained_model_file_path
            logging.info(f"Loading model for evaluation :{self.model_trainer_artifact.trained_model_file_path}")
            trained_model_object = load_object(file_path=trained_model_file_path)
            #creating training and test file path
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            #creating schema file path
            schema_file_path = self.data_validation_artifact.schema_file_path
            #Loading train and test dataframe from the file
            train_dataframe= load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_dataframe = load_data(file_path=test_file_path,schema_file_path=schema_file_path)
            #creating schema dictionary
            schema_content = read_yaml_file(file_path=schema_file_path)
            #specifying target column
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            #Converting target column in numpy array
            logging.info(f"Converting target column in numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            #dropping target column from the dataframe
            logging.info(f"Dropping target column from the DataFrame")
            train_dataframe.drop(target_column_name,axis=1,inplace=True)
            test_dataframe.drop(target_column_name,axis=1,inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")
            #creating best model instance
            model=self.get_best_model()

            if model is None :
                logging.info(f"Not found any existing model.Hence accepting trained model")
                model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=True,
                                evaluated_model_path=trained_model_file_path)
                
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted.Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            #creating list of model 
            model_list =[model,trained_model_object]
            #creating model evaluation artifact 
            metric_info_artifact = evaluate_classification_model(model_list=model_list,
                                                X_train=train_dataframe,
                                                y_train=train_target_arr,
                                                X_test=test_dataframe,
                                                y_test=test_target_arr,
                                                base_accuracy=self.model_trainer_artifact.model_accuracy
                                                )    
            logging.info(f"Model evaluation completed. Model metric artifact:{metric_info_artifact}")
            
            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                        evaluated_model_path=trained_model_file_path)
                logging.info(response)
                return response

            if metric_info_artifact.index_number==1:
                model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=True,
                                            evaluated_model_path=trained_model_file_path)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model evaluation artifact{model_evaluation_artifact} created")
            else:
                logging.info(f"Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                is_model_accepted=False)
            
            return model_evaluation_artifact

        except Exception as e:
            raise CreditException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'>>'*30} Model Evaluation log completed.{'<<'*20}\n")
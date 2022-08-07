import sys
from credit.exception import CreditException
from credit.entity.config_entity import DataIngestionConfig, DataValidationConfig, \
     DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig,\
        TrainingPipelineConfig
from credit.util import read_yaml_file
from credit.constants import *
from credit.logger import logging
from datetime import datetime

class configuration:
    def __init__(self,config_file_path:str=CONFIG_FILE_PATH,
                    current_timestamp:str=CURRENT_TIME_STAMP) -> None:
        """Base Class for configuration for components of the pipeline"""
        try:
            self.config_info= read_yaml_file(file_path=config_file_path)
            self.timestamp= current_timestamp
            self.training_pipeline_config= self.get_training_pipeline_config()
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_ingestion_config(self)->DataIngestionConfig:
        """This method returns DataIngestionConfig(tuple) which has values
            "dataset_download_url"(str): url of the dataset to download 
            ,"zip_download_dir"(str): directory in which the downloaded dataset zipped file
            ,"raw_data_dir"(str): directory in which extracted dataset is stored
            ,"ingested_train_dir"(str): directory for ingested data training set 
            ,"ingested_test_dir"(str): directory for ingested test set"""
        try:
            
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            #creating path for data_ingestion artifact directory
            data_ingestion_artifact_dir=os.path.join(
                                        artifact_dir,
                                        DATA_INGESTION_ARTIFACT_DIR,
                                        self.timestamp
                                         )

            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            
            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            
            # creating path for directory in which the downloaded dataset zipped file
            zip_download_dir = os.path.join(data_ingestion_artifact_dir,
                                    data_ingestion_info[DATA_INGESTION_ZIP_DOWNLOAD_DIR_KEY]
                                            )
            # creating path for directory in which extracted dataset is stored                               
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
                                        )
            # creating path for directory ingested data
            ingested_data_dir = os.path.join(data_ingestion_artifact_dir,
                                    data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
                                            )
            # creating path for directory ingested data training dataset                               
            ingested_train_dir = os.path.join(ingested_data_dir,
                                    data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]
                                         )
            # creating path for directory ingested data testing dataset                             
            ingested_test_dir =os.path.join(ingested_data_dir,
                                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]
                                        )

            # creating data_ingestion_config
            data_ingestion_config=DataIngestionConfig(dataset_download_url=dataset_download_url, 
                                    zip_download_dir=zip_download_dir, 
                                    raw_data_dir=raw_data_dir, 
                                    ingested_train_dir=ingested_train_dir, 
                                    ingested_test_dir=ingested_test_dir
                                        )

            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_validation_config(self)->DataValidationConfig:
        """This method returns DataValidationConfig(tuple) which has values
            "schema_file_path"(str):path for schema.yaml,
            "report_file_path"(str):path where data drift report to be stored,
            "report_page_file_path(str):path where data drift report page to be stored"
            """
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            #creating path for data_validation_artifact directory
            data_validation_artifact_dir=os.path.join(
                                        artifact_dir,
                                        DATA_VALIDATION_ARTIFACT_DIR_NAME,
                                        self.timestamp
                                        )

            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]

            #creating schema file path
            schema_file_path = os.path.join(CURR_DIR,
                                data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                            data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
                                    )
            #creating report file path
            report_file_path = os.path.join(data_validation_artifact_dir,
                            data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
                                )
            #creating report page file path 
            report_page_file_path = os.path.join(data_validation_artifact_dir,
                            data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]

                                    )

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path,
            )
            return data_validation_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_transformation_config(self)->DataTransformationConfig:
        """This method returns DataTransformationConfig(tuple) which has values
            "transformed_train_dir"(str): dir transformed train dataset ,
            "transformed_test_dir"(str): dir transformed test dataset,
            "preprocessed_object_file_path"(str): preprocessed object file path
            """
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            #creating data_transformation_artifact
            data_transformation_artifact_dir=os.path.join(
                                        artifact_dir,
                                        DATA_TRANSFORMATION_ARTIFACT_DIR,
                                        self.timestamp
                                            )

            data_transformation_config_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            #creating preprocessed object file path
            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
                                                )

            #creating transformed training test directory
            transformed_train_dir=os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
                                                )

            #creating transformed testing test directory
            transformed_test_dir = os.path.join(
                    data_transformation_artifact_dir,
                    data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
                    data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]

                        )
            
            #creating data_tranformation_config
            data_transformation_config=DataTransformationConfig(
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir
                                    )

            logging.info(f"Data transformation config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_trainer_config(self)->ModelTrainerConfig:
        """This method returns ModelTrainerConfig(tuple) which has values
        "trained_model_file_path"(str): trained model object file path,
        "base_accuracy"(float): Accuracy above which model will  be accepted ,
        "model_config_file_path(str): model configuration yaml file path"""
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            #creating model trainer artifact dir
            model_trainer_artifact_dir = os.path.join(
                artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR ,
                self.timestamp)
            
            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            #creating trained model file path 
            trained_model_file_path = os.path.join(model_trainer_artifact_dir,
                                    model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],
                                    model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY])
            #defining base accuracy
            base_accuracy= model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            #creating model config file path
            model_config_file_path=os.path.join(model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                                    model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY])
            #creating model trainer config
            model_trainer_config = ModelTrainerConfig(
                                trained_model_file_path=trained_model_file_path,
                                base_accuracy=base_accuracy,
                                model_config_file_path=model_config_file_path)
            
            logging.info(f"Model trainer config:{model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        """This method returns ModelEvaluationConfig(tuple) which has values
       "model_evaluation_file_path"(str),
       "time_stamp" """
        try:
           
            model_evaluation_config_info= self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            
            artifact_dir =self.training_pipeline_config.artifact_dir
            #creating model evaluation artifact dir
            model_evaluation_artifact_dir= os.path.join(artifact_dir,
                            MODEL_EVALUATION_ARTIFACT_DIR)
            #creating model evaluation file path
            model_evaluation_file_path = os.path.join(model_evaluation_artifact_dir,
                                        model_evaluation_config_info[MODEL_EVALUATION_FILE_NAME_KEY])
            #creating model evaluation config
            model_evaluation_config = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                            time_stamp=self.timestamp)

            logging.info(f"Model evaluation config:{model_evaluation_config}")
            return model_evaluation_config            
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            time_stamp=f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config_info=self.config_info[MODEL_PUSHER_CONFIG_KEY]
            #creating model pusher dir/saved models file path
            export_dir_path = os.path.join(CURR_DIR,model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                            time_stamp)
            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config{model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        """This method returns the TrainingPipelineConfig(tuple) which has value
        "artifact_dir"(str): directory where the artifacts of each component of the pipeline are located"""
        try:
            training_pipeline_config_key=self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            
            # creating path for artifact directory
            artifact_dir=os.path.join(CURR_DIR,
                            training_pipeline_config_key[TRAINING_PIPELINE_NAME_KEY],
                            training_pipeline_config_key[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            
            training_pipeline_config=TrainingPipelineConfig(artifact_dir=artifact_dir)
            
            logging.info(f"Training pipleine config: {training_pipeline_config}")
            return training_pipeline_config

        except Exception as e:
            raise CreditException(e,sys) from e

    
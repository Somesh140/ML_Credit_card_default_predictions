import sys
from credit.exception import CreditException
from credit.entity.config_entity import DataIngestionConfig, DataValidationConfig, \
     DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig,\
        TrainingPipelineConfig
from credit.util import read_yaml_file
from credit.constants import *
from credit.logger import logging

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
        try:
            pass
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            pass
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        try:
            pass
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            pass
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

    
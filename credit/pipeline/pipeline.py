import os,sys
from datetime import datetime
from threading import Thread
import uuid
import pandas as pd
from credit.exception import CreditException
from credit.logger import logging
from credit.entity.artifact_entity import *
from credit.entity.config_entity import *
from credit.component.data_ingestion import DataIngestion
from credit.component.data_validation import DataValidation
from credit.component.data_transformation import DataTransformation
from credit.component.model_trainer import ModelTrainer
from credit.component.model_evaluation import ModelEvaluation
from credit.component.model_pusher import ModelPusher
from credit.config.configuration import configuration
from credit.util import get_current_timestamp
from credit.constants import *


class Pipeline(Thread):
    experiment : Experiment = Experiment(*([None]*11))

    experiment_file_path = None
    
    
    def __init__(self,config:configuration=configuration()) -> None:
        try:
            #creating artifact directory
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            #creating training experiment file path
            Pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False, name="pipeline")
            self.config=config
        except Exception as e:
            raise CreditException(e,sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                            data_ingestion_config=self.config.get_data_ingestion_config())
            
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CreditException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) \
            -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                            data_validation_config=self.config.get_data_validation_config(),
                            data_ingestion_artifact=data_ingestion_artifact
                                   )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CreditException(e, sys) from e


    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CreditException(e, sys)


    def start_model_trainer(self,
                        data_transformation_artifact:DataTransformationArtifact,
                        )->ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                        model_trainer_config=self.config.get_model_trainer_config(),
                        data_transformation_artifact=data_transformation_artifact)
            
            return model_trainer.initiate_model_trainer()        
        except Exception as e:
            raise CreditException(e,sys) from e

    def start_model_evaluation(self,data_ingestion_artifact:DataIngestionArtifact,
                            data_validation_artifact:DataValidationArtifact,
                            model_trainer_artifact:ModelTrainerArtifact
                            )->ModelEvaluationArtifact:
        try:
            model_eval =    ModelEvaluation(
                            model_evaluation_config= self.config.get_model_evaluation_config(),
                            data_ingestion_artifact= data_ingestion_artifact,
                            data_validation_artifact=data_validation_artifact,
                            model_trainer_artifact= model_trainer_artifact)
            return model_eval.initiate_model_evaluation()

        except Exception as e:
            raise CreditException(e,sys) from e

    def start_model_pusher(self,
                    model_eval_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact
                )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CreditException(e,sys) from e

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment

            logging.info("Pipeline starting")
            experiment_id = str(uuid.uuid4())
            running_status =True

            Pipeline.experiment = Experiment(experiment_id=experiment_id,
                                            Initialization_timestamp=self.config.timestamp,
                                            artifact_time_stamp=self.config.timestamp,
                                            running_status=running_status,
                                            start_time=datetime.now(),
                                            stop_time=None,
                                            execution_time=None,
                                            experiment_file_path=Pipeline.experiment_file_path,
                                            is_model_accepted=None,
                                            message="Pipeline has been started",
                                            accuracy=None)

            logging.info(f":Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
            
            
            
            # data ingestion
            data_ingestion_artifact=self.start_data_ingestion()

            #data validation
            data_validation_artifact=self.start_data_validation(
                                        data_ingestion_artifact=data_ingestion_artifact)

            #data transformation
            data_transformation_artifact=self.start_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    data_validation_artifact=data_validation_artifact
            )

            #Model training
            model_trainer_artifact=self.start_model_trainer(
                        data_transformation_artifact=data_transformation_artifact)

            #Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            
            #Model Pusher
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f"Model Pusher artifact:{model_pusher_artifact}")
            else:
                logging.info("Trained model rejected")
            logging.info("Pipeline completed.")

            stop_time = datetime.now()
            running_status=False
            
            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
                                             Initialization_timestamp=self.config.timestamp,
                                             artifact_time_stamp=self.config.timestamp,
                                             running_status=running_status,
                                             start_time=Pipeline.experiment.start_time,
                                             stop_time=stop_time,
                                             execution_time=stop_time - Pipeline.experiment.start_time,
                                             message="Pipeline has been completed.",
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=model_evaluation_artifact.is_model_accepted,
                                             accuracy=model_trainer_artifact.model_accuracy
                                             )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()

        except Exception as e:
            raise CreditException(e,sys) from e
    
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise CreditException(e,sys) from e



    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                #creating dictionary from named tuple
                experiment_dict = experiment._asdict()
                #separating key value pair in the dictionary
                experiment_dict: dict = {key:[value] for key,value in experiment_dict.items()}
                #updating created timestamp and experiment file path calue in dictionary
                experiment_dict:dict.update({
                    "created_time_stamp":[get_current_timestamp()],
                    "experiment_file_path":[os.path.basename(Pipeline.experiment.experiment_file_path)]
                    })
                #creating dataframe from dictionary
                experiment_report = pd.DataFrame(experiment_dict)
                #creating sirectory to save the training experiment file
                os.makedirs(os.path.dirname(Pipeline.experiment_file_path),exist_ok=True)
                #if path exists then appending the existing file else creating a new file
                if os.path.exists(Pipeline.experiment_file_path):
                    #saving dataframe to csv file
                    experiment_report.to_csv(Pipeline.experiment_file_path,
                                            index=False,header=False,mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path,
                                            index=False,header=True,mode="w")
            else:
                logging.info(f"First start experiment")
        except Exception as e:
            raise CreditException(e,sys) from e

    @classmethod
    def get_experiments_status(cls,limit:int=5)-> pd.DataFrame:
        """This method gets last five records of training history from experiment.csv"""
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit=-1*int(limit)
                return df[limit:].drop(columns=["experiment_file_path","Initialization_timestamp"],axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise CreditException(e,sys) from e
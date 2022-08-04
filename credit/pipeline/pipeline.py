import os,sys
from credit.exception import CreditException
from credit.logger import logging
from credit.entity.artifact_entity import *
from credit.entity.config_entity import *
from credit.component.data_ingestion import DataIngestion
from credit.component.data_validation import DataValidation
from credit.config.configuration import configuration


class Pipeline:
    def __init__(self,config:configuration=configuration()) -> None:
        try:
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
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CreditException(e, sys) from e

    def run_pipeline(self):
        try:
            # data ingestion
            data_ingestion_artifact=self.start_data_ingestion()

            #data validation
            data_validation_artifact=self.start_data_validation(
                                        data_ingestion_artifact=data_ingestion_artifact)
            
        except Exception as e:
            raise CreditException(e,sys) from e
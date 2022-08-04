import sys,os
from credit.exception import CreditException
from credit.logger import logging
from credit.entity.config_entity import DataIngestionConfig
from credit.entity.artifact_entity import DataIngestionArtifact
from zipfile import ZipFile
from six.moves import urllib
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"\n\n{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CreditException(e,sys) from e

    def download_credit_data(self,) -> str:
        """This method downloads the dataset zipped file from the url"""
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            zip_download_dir = self.data_ingestion_config.zip_download_dir
            
            # creating zip_download_dir
            os.makedirs(zip_download_dir,exist_ok=True)

            credit_file_name = os.path.basename(download_url)
            
            #creating zip file path
            zip_file_path = os.path.join(zip_download_dir,credit_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{zip_file_path}]")
            
            #downloading file from the url and storing in zip_file_path
            urllib.request.urlretrieve(download_url,zip_file_path)
            logging.info(f"File :[{zip_file_path}] has been downloaded successfully.")
            return  zip_file_path

        except Exception as e:
            raise CreditException(e,sys) from e

    def extract_zip_file(self,zip_file_path:str):
        """This method extracts the dataset from zipped file and stores in raw_data directory"""
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            #creating raw_data directory
            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting zip file: [{zip_file_path}] into dir: [{raw_data_dir}]")
            
            #Extracting zip file
            with ZipFile(zip_file_path) as credit_zip_obj:
               credit_zip_obj.extractall(path=raw_data_dir)
            #shutil.unpack_archive(zip_file_path, raw_data_dir)
            logging.info(f"Extraction completed")
        
        except Exception as e:
            raise CreditException(e,sys) from e

    def split_data_train_test(self)->DataIngestionArtifact:
        """This method splits the raw data into training and test dataset"""
        try:    
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            is_ingested=False
            # selecting file from raw_data directory
            file_name = os.listdir(raw_data_dir)[0]
            #creating file path for credit csv file
            credit_file_path = os.path.join(raw_data_dir,file_name)
            logging.info(f"Reading csv file: [{credit_file_path}]")
            credit_data_frame = pd.read_csv(credit_file_path)

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            #using StratifiedShufflesplit splitting the data based on target column default.payment.next.month
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            #splitting data through train_indedx and test_index
            for train_index,test_index in split.split(credit_data_frame, credit_data_frame["default.payment.next.month"]):
                strat_train_set = credit_data_frame.loc[train_index]
                strat_test_set = credit_data_frame.loc[test_index]
            
            #creating training dataset filepath
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)
            #creating testing dataset filepath
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            #Saving split training set csv file
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)
                is_ingested=True
            
            #Saving split testing set csv file
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
                is_ingested=True
            
            #data ingestion status
            if is_ingested==True:
                message=f"Data ingestion completed successfully."
            else:
                message=f"Data ingestion unsuccessful."


            #creating data_ingestion_artifact
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=is_ingested,
                                message=message
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise CreditException(e,sys) from e
    
    
    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        """This method initiates the data ingestion & returns DataIngestionArtifact(tuple) which has values
         "train_file_path"(str):path for training dataset,
         "test_file_path"(str):path for testing dataset,
         "is_ingested"(bool):True or False,
         "message"(str):Data ingestion completed successfully./Data ingestion unsuccessful."""
        try:
            #downloading dataset zip file 
            zip_file_path =  self.download_credit_data()
            #extracting zip file
            self.extract_zip_file(zip_file_path=zip_file_path)
            #returning data ingestion artifact
            return self.split_data_train_test()

        except Exception as e:
            raise CreditException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'>>'*20}Data Ingestion log completed.{'<<'*20} \n")
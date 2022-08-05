from credit.entity.config_entity import *
from credit.entity.artifact_entity import *
import os,sys,json
import pandas as pd
from credit.logger import logging
from credit.exception import CreditException
from credit.config.configuration import configuration
from credit.constants import *
from credit.util import read_yaml_file
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile



class DataValidation:

    def __init__(self,data_validation_config:DataValidationConfig,
                data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"\n\n{'>>'*30}Data Validation log started.{'>>'*30}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact

        except Exception as e:
            raise CreditException(e,sys) from e 

    def get_train_and_test_df(self)->pd.DataFrame:
        """This method gets training and testing dataframe"""
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df,test_df
        except Exception as e:
            raise CreditException(e,sys) from e


    def is_train_test_file_exists(self)->bool:
        """This method checks if training and testing file exists"""
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            #checking if file exists
            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path) 

            is_available =  is_test_file_exist and is_train_file_exist
            logging.info(f"Is train and test file exists?->{is_available}")
            
            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message = f"Training file:{training_file} or Testing file: {testing_file}" \
                            "is not present"
                raise Exception(message)    
            
            return is_available

        except Exception as e:
            raise CreditException(e,sys) from e

    def validate_dataset_schema(self)->bool:
        """This method performs checks on the train and test dataset to validate it using 
        schema file"""
        try:
            validation_status=False
            # validate training and testing dataset using schema file
            train_df,test_df=self.get_train_and_test_df() 
            # checking number of columns in training dataset 
            chk1 = self.check_length_column(train_df)              
            logging.info(f"Number of columns in training dataset match with the schema:{chk1}")
           # checking number of columns in testing dataset 
            chk2 = self.check_length_column(test_df)
            logging.info(f"Number of columns in testing dataset match with the schema:{chk2}")
                           
            # Check column names in training dataset
            chk3= self.check_column_names(train_df)
            logging.info(f"all columns in training dataset are correct: {chk3}")
            # Check column names in training dataset    
            chk4= self.check_column_names(test_df)
            logging.info(f"all columns in testing dataset are correct: {chk4}")
            

            if (chk1 and chk2  and chk3 and chk4) is False:
                validation_status =False
                raise Exception(f"data is not valid")

            else:
                validation_status =True
              
            return validation_status
        except Exception as e:
            raise CreditException(e,sys) from e
    
    def check_length_column(self,df:pd.DataFrame)->bool:
        """This method compares number of columns in df with those in schema file"""
        try:
            schema = read_yaml_file(self.data_validation_config.schema_file_path)
            if len(df.columns)==len(schema[DATASET_SCHEMA_COLUMNS_KEY].keys()):
                logging.info(f"No. of columns in dataset :[{len(df.columns)}] and number of columns in schema file [{len(schema[DATASET_SCHEMA_COLUMNS_KEY].keys())}]")
                chk = True
            else:
                logging.info(f"No. of columns in datset :[{len(df.columns)}] and number of columns in schema file [{len(schema[DATASET_SCHEMA_COLUMNS_KEY].keys())}]")
                chk=False
            return chk
        except Exception as e:
            raise CreditException(e,sys) from e
 

    def check_column_names(self,df:pd.DataFrame)->bool:
        """This method checks whether the column name in dataframe is same
        as that in the schema file"""
        try:
            schema = read_yaml_file(self.data_validation_config.schema_file_path)
            columns_names = list(df.columns)
            
            for col in columns_names:
                if col not in list(schema[DATASET_SCHEMA_COLUMNS_KEY].keys()):
                    logging.info(f"{col} not in the dataset")
                    chk=False
                    break
                else:
                    chk = True         
            return chk
        except Exception as e:
            raise CreditException(e,sys) from e

    def save_data_drift_report_page(self):
        """This method creates data drift html page"""
        try:
            #creating dashboard object
            dashboard = Dashboard(tabs=[DataDriftTab()])
            #getting dataset for comparison
            train_df,test_df = self.get_train_and_test_df()
            #calculate datadrift
            dashboard.calculate(train_df,test_df)
            #creating report page file path           
            report_page_file_path = self.data_validation_config.report_page_file_path
            #creating report page dir path
            report_page_dir = os.path.dirname(report_page_file_path)
            #creating report page dir
            os.makedirs(report_page_dir,exist_ok=True)
            #saving dashboard objecty to report page file path 
            dashboard.save(report_page_file_path)
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_drift_report(self):
        """This method gets datadrift json report"""
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            #getting 2 datasets for comparison
            train_df,test_df=self.get_train_and_test_df()
            #calculating Data drift 
            profile.calculate(train_df,test_df)
            
            #creating report file path
            report_file_path = self.data_validation_config.report_file_path
            #creating dir path for the report
            report_dir = os.path.dirname(report_file_path)
            #creating directory for report dir
            os.makedirs(report_dir,exist_ok=True)    
            #loading report in json format
            report = json.loads(profile.json())
            #dumping json object in the file
            with open(report_file_path,"w") as report_file:
                json.dump(report,report_file,indent=6)

            return report    

        except Exception as e:
            raise CreditException(e,sys) from e
    
    
    
    def is_data_drift_found(self)->bool:
        try:
            report = self.get_data_drift_report()
            #check whether data drift            
            status=report[DATA_DRIFT_KEY][DATA_KEY][METRICS_KEY][DATASET_DRIFT_KEY]

            logging.info(f"data_drift : {status}")
            self.save_data_drift_report_page()
            return status
        except Exception as e:
            raise CreditException(e,sys) from e

    
    def initiate_data_validation(self)->DataValidationArtifact:
        """This method initiates the data validation & returns DataValidationArtifact(tuple) 
        which has values
        "schema_file_path"(str): schema file path ,
        "report_file_path"(str):report file path,
        "report_page_file_path"(str): report page file path,
        "is_validated"(bool):True/False,
        "message(str):"Data Validation performed successfully"/"Data Validation unsuccessful"
         """
        try:
            self.is_train_test_file_exists()
            is_validated=self.validate_dataset_schema()
            drift= self.is_data_drift_found()
            logging.info(f"data_drift : {drift}")
            if (is_validated==True and drift==False):
                message="Data Validation performed successfully"

            else:
                message="Data Validation unsuccessful"
            
            #creating data_validation_artifact
            data_validation_artifact=DataValidationArtifact(
                                    schema_file_path=self.data_validation_config.schema_file_path,
                                    report_file_path=self.data_validation_config.report_file_path,
                                    report_page_file_path=self.data_validation_config.report_page_file_path,
                                    is_validated=is_validated,
                                    message=message)
            logging.info(f"Data validation artifact:{data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CreditException(e,sys) from e 

    def __del__(self):
        logging.info(f"\n{'>>'*20}Data Validation log completed.{'>>'*20}\n")       
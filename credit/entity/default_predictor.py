import os,sys
from credit.logger import logging
from credit.exception import CreditException
from credit.util import load_object

import pandas as pd

class CreditData:
    def __init__(self,
                ID: int,
                LIMIT_BAL: float,
                SEX: int,
                EDUCATION: int,
                MARRIAGE: int,
                AGE: int,
                PAY_0: int,
                PAY_2: int,
                PAY_3: int,
                PAY_4: int,
                PAY_5: int,
                PAY_6: int,
                BILL_AMT1: float,
                BILL_AMT2: float,
                BILL_AMT3: float,
                BILL_AMT4: float,
                BILL_AMT5: float,
                BILL_AMT6: float,
                
                PAY_AMT1: float,
                PAY_AMT2: float,
                PAY_AMT3: float,
                PAY_AMT4: float,
                PAY_AMT5: float,
                PAY_AMT6: float,
                
                default_payment_next_month: int=None):
        try:
            self.ID = ID
            self.AGE= AGE
            self.BILL_AMT1=BILL_AMT1
            self.BILL_AMT2= BILL_AMT2
            self.BILL_AMT3= BILL_AMT3
            self.BILL_AMT4= BILL_AMT4
            self.BILL_AMT5= BILL_AMT5
            self.BILL_AMT6= BILL_AMT6
            self.EDUCATION= EDUCATION
            self.LIMIT_BAL=LIMIT_BAL
            self.MARRIAGE=MARRIAGE
            self.PAY_0=PAY_0
            self.PAY_2=PAY_2
            self.PAY_3=PAY_3
            self.PAY_4=PAY_4
            self.PAY_5=PAY_5
            self.PAY_6=PAY_6
            self.PAY_AMT1=PAY_AMT1
            self.PAY_AMT2=PAY_AMT2
            self.PAY_AMT3=PAY_AMT3
            self.PAY_AMT4=PAY_AMT4
            self.PAY_AMT5=PAY_AMT5
            self.PAY_AMT6=PAY_AMT6
            self.SEX=SEX
            self.default_payment_next_month=default_payment_next_month
            
            
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_default_data_as_dict(self):
        try:
            input={
                "ID":[self.ID],
                
                "LIMIT_BAL": [self.LIMIT_BAL],
                "SEX": [self.SEX],
                "EDUCATION": [self.EDUCATION],
                
                "MARRIAGE": [self.MARRIAGE],
                "AGE": [self.AGE],
                "PAY_0": [self.PAY_0],
                "PAY_2": [self.PAY_2],
                "PAY_3": [self.PAY_3],
                "PAY_4": [self.PAY_4],
                "PAY_5": [self.PAY_5],
                "PAY_6": [self.PAY_6],
                "BILL_AMT1":[self.BILL_AMT1],
                "BILL_AMT2": [self.BILL_AMT2],
                "BILL_AMT3": [self.BILL_AMT3],
                "BILL_AMT4": [self.BILL_AMT4],
                "BILL_AMT5": [self.BILL_AMT5],
                "BILL_AMT6": [self.BILL_AMT6],
                
                
                "PAY_AMT1": [self.PAY_AMT1],
                "PAY_AMT2": [self.PAY_AMT2],
                "PAY_AMT3": [self.PAY_AMT3],
                "PAY_AMT4": [self.PAY_AMT4],
                "PAY_AMT5": [self.PAY_AMT5],
                "PAY_AMT6": [self.PAY_AMT6],
                
                "default.payment.next.month": [self.default_payment_next_month]

            }
            
            
            return input
        except Exception as e:
            raise CreditException(e,sys) from e
    
    
    def get_default_input_dataframe(self)->pd.DataFrame:
        """This method converts imput dictionary to pandas dataframe"""
        try:
            default_input_dict = self.get_default_data_as_dict()
            return pd.DataFrame(default_input_dict)
        except Exception as e:
            raise CreditException(e,sys) from e



class DefaultPredictor:
    def __init__(self,model_dir:str):
        try:
            #initialise with saved model dir
            self.model_dir = model_dir 
        except Exception as e:
           raise CreditException(e,sys) from e

    def get_latest_model_path(self)->str:
        """This method is used to get the latest model path"""
        try:
            #getting the dir names 
            folder_name =list(map(int,os.listdir(self.model_dir)))
            #getting the latest model dir
            latest_model_dir= os.path.join(self.model_dir,f"{max(folder_name)}")
            #getting the latest file name
            file_name = os.listdir(latest_model_dir)[0]
            #creating the latest model file path
            latest_model_path = os.path.join(latest_model_dir,file_name)
            return latest_model_path
        except Exception as e:
            raise CreditException(e,sys) from e


    def predict(self,X):
        """This method is used to predict default"""
        try:
            #getting model file path
            model_path = self.get_latest_model_path()
            #loading model object
            model = load_object(file_path=model_path)
            default_payment_next_month = model.predict(X)
            return default_payment_next_month[0]
        except Exception as e:
            raise CreditException(e,sys) from e
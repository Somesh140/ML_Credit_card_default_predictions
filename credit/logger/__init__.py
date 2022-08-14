import os
import logging
from credit.constants import LOG_DIR
import pandas as pd

from credit.util import get_current_timestamp

def log_filename()->str:
    "This function return name for the log file based oin current timestamp"
    return  f"log_{get_current_timestamp()}.log"


log_file_name=log_filename()

#checking if log directory exists and creating if not
os.makedirs(LOG_DIR,exist_ok=True)

#creating log file path
log_file_path=os.path.join(LOG_DIR,log_file_name)

#configuring log file
logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    format="[%(asctime)s]-[%(levelname)s]-[%(lineno)s]-[%(filename)s]-[%(funcName)s()]-[%(message)s]",
                    level=logging.INFO)

def get_log_df(filepath):
    "This functions reads the log file return a dataframe of the inforamtion from the log file"
    data=[]
    with open(filepath) as log_file:
        for line in log_file.readlines():
            data.append(line.split("]-["))
    
    log_df=pd.DataFrame(data)
    columns=["Timestamp","Log_level","Line_Number","Filename","FunctionName","Message"]
    log_df.columns=columns
    log_df["log_message"] = log_df['Timestamp'].astype(str) +":$"+ log_df["Message"]

    return log_df[["log_message"]]





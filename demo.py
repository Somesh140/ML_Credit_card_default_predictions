from credit.config.configuration import configuration
from credit.pipeline.pipeline import Pipeline
from credit.exception import CreditException
from credit.logger import logging

import os,sys

def main():
    try:
        #config_path=os.path.join("config","config.yaml")
        #pipeline = Pipeline(configuration(config_file_path=config_path))
        #pipeline.start()
        
        pipeline = Pipeline()
        pipeline.run_pipeline()
        logging.info("main function execution completed")
        #data_transformation_config = Configuartion().get_data_transformation_config()
        #print(data_transformation_config)
        #schema_file_path = r"D:\Datascience_Projects\ML_project_1\config\schema.yaml"
        #file_path = r"D:\Datascience_Projects\ML_project_1\credit\artifact\data_ingestion\2022-07-10-16-12-33\ingested_data\train\credit.csv" 

        #df=DataTransformation.load_data(file_path=file_path,schema_file_path=schema_file_path)

        #print(df.columns)
        #print(df.dtypes)

    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__=="__main__":
    main()
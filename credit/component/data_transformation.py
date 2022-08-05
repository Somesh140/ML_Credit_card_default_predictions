from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from credit.util import *
from sklearn.base import BaseEstimator,TransformerMixin
from credit.logger import logging
from credit.exception import CreditException
from credit.config.configuration import *
from credit.entity.artifact_entity import *
from credit.constants import *

class FeatureDropper(BaseEstimator, TransformerMixin):

    def __init__(self,columns):
        """
        FeatureDropper Initialization
        columns: Feature we need to drop
        """
        try:
            self.columns = columns
            
        except Exception as e:
            raise CreditException(e, sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            return X.drop(self.columns,axis=1)
        except Exception as e:
            raise CreditException(e,sys) from e



class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig,
                data_ingestion_artifact:DataIngestionArtifact,
                data_validation_artifact:DataValidationArtifact):
        try:
            logging.info(f"\n\n{'>>'*30}Data Transforamtion log started.{'>>'*30}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_transformer_object(self)->ColumnTransformer:
        """This method returns tranformed object from which feature is dropped
        and standard scaling is performed"""
        try:
            #creating schema file path
            schema_file_path = self.data_validation_artifact.schema_file_path
            #reading schema file path
            dataset_schema = read_yaml_file(file_path=schema_file_path)
            # identifying numerical columns
            numerical_columns=dataset_schema[NUMERICAL_COLUMN_KEY]
            
            #feature that needs to dropped in transformation
            drop_feature= dataset_schema[DROP_COLUMN_KEY]
            #creating pipeline
            num_pipeline = Pipeline(steps=[
                        ('featuredropper',FeatureDropper(columns=drop_feature)),
                        ('scaling',StandardScaler())
                        ])
            logging.info(f"Numerical Columns:{numerical_columns}")
            #creating preprocessing object
            preprocessing = ColumnTransformer([
                            ('num_pipeline',num_pipeline,numerical_columns),
                                         ])
            return preprocessing

        except Exception as e:
            raise CreditException(e,sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        """This method initiates the data transformation & returns DataTransformationArtifact(tuple) 
        which has values 
        "is_transformed"(bool):True/False, 
        "message"(str):"Data Transformation Successful"/"Data Transformation unSuccessful", 
        "transformed_train_file_path"(str):path of transformed train file,
        "transformed_test_file_path"(str):path of transformed test file,
        "preprocessed_object_file_path"(str):
        """
        try:
            #obtaining preprocessing object
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file path.")
            #creating path for train file 
            train_file_path =self.data_ingestion_artifact.train_file_path
            #creating path for test file 
            test_file_path = self.data_ingestion_artifact.test_file_path
            #creating schema file path
            schema_file_path= self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data  as pandas dataframe.")
            #Loading training data  as pandas dataframe
            train_df = load_data(file_path=train_file_path,
                                schema_file_path=schema_file_path)
            #Loading training data  as pandas dataframe
            test_df = load_data(file_path=train_file_path,
                                schema_file_path=schema_file_path)
            #reading schema file 
            schema = read_yaml_file(file_path=schema_file_path)
            #Specifying target feature column
            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            #concatenating preprocessed input feature train array with target feature train df 
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            #concatenating preprocessed input feature test array with target feature test df 
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            #creating tranforemed train and test dir
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            #creating train and test file name path
            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")
            #creating transformed train and test file path
            transformed_train_file_path = os.path.join(transformed_train_dir,train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir,test_file_name)    
            #saving numpy array data
            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)
            #creating preprocessed object file path
            preprocessing_obj_file_path=self.data_transformation_config.preprocessed_object_file_path
            #saving preprocessing object
            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)
            #creating data transformation artifact
            data_transformation_artifact= DataTransformationArtifact(is_transformed=True,
                            message="Data Transformation Successful",
                            transformed_train_file_path=transformed_train_file_path,
                            transformed_test_file_path=transformed_test_file_path,
                            preprocessed_object_file_path=preprocessing_obj_file_path
            )
            logging.info(f"Data transformation artifact:{data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e,sys) from e



    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'>>'*30}\n\n")

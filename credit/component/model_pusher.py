import shutil
from credit.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from credit.entity.config_entity import ModelPusherConfig
from credit.exception import CreditException
from credit.logger import logging
import os,sys
from credit.constants import *

class ModelPusher:
    
    def __init__(self,model_pusher_config:ModelPusherConfig,
                model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            logging.info(f"\n\n{'>>'*30} Model Pusher log started{'<<'*30}")
            self.model_pusher_config= model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise CreditException(e,sys) from e

    def initiate_model_pusher(self)->ModelPusherArtifact:
        """This method creates saved model directory where the best evaluated model that is to
        be used for production is to be kept

        ================================================\n
        Returns ModelPusherArtifact(tuple)
        "is_model_pushed", "export_model_file_path" """
        try:
            #creating evaluated model path
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            #creating export directory file path
            export_dir = self.model_pusher_config.export_dir_path
            #reading model file name from evaluated model file path
            model_file_name = os.path.basename(evaluated_model_file_path)
            #creating file path for export model
            export_model_file_path = os.path.join(export_dir,model_file_name)
            logging.info(f"Exporting model file:[{export_model_file_path}]")
            os.makedirs(export_dir,exist_ok=True)
            #creating copy of evaluated model in export model file path
            shutil.copy(src=evaluated_model_file_path,dst=export_model_file_path)
            
            logging.info(f"Trained model:{evaluated_model_file_path} is copied in export dir:{export_model_file_path}")
            #creating model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(is_model_pushed=True,
                                            export_model_file_path=export_model_file_path)
            logging.info(f"Model Pusher artifact:[{model_pusher_artifact}]")

            return model_pusher_artifact
        except Exception as e:
            raise CreditException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*20} Model Pusher log completed.{'<<'*20}\n\n")
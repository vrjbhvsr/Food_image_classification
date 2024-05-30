from Food_Classification.components.data_transformation import DataTransformation
from Food_Classification.components.tensorboard import preparetensorboard
from Food_Classification.components.prepare_base_model import PrepareBaseModel
from Food_Classification.config.configuration import ConfigurationManager
from Food_Classification.components.model_trainer import Model_Training
from Food_Classification.components.data_ingestion import DataIngestion
from Food_Classification.components.Model_Evaluation import Model_Evaluation
from Food_Classification.components.model_pusher import ModelPusher
from Food_Classification import logger
import torch


class TrainingPipeline:
    def __init__(self,config):
        self.config = config()

    def dataingestionpipeline(self):
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config = data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.unzip_file()
        except Exception as e:
            raise e
        

    def basemodelpipeline(self):
        try:
            prepare_base_model_config = self.config.get_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()

        except Exception as e:
            raise e
        

    def datatransformationpipeline(self):
        try:
            transformation_config = self.config.get_DataTransformConfig()
            transform = DataTransformation(config=transformation_config)
            data_transformation_artifact = transform.initiate_datatransfrom()
            return data_transformation_artifact
        except Exception as e:
            raise e
        

    def trainingpipeline(self,artifact):
        try:
            train_config = self.config.get_training_config()
            trainig = Model_Training(config=train_config, transformation_Artifacts=artifact)
            training_artifacts = trainig.initiate_Model_training()
            return training_artifacts
        except Exception as e:
            raise e
        

    def evaluationpipeline(self,training_artifact, transformation_artifact):
        try:
            evaluation_config = self.config.get_evaluation_config()
            evaluation = Model_Evaluation(config=evaluation_config, training_artifact = training_artifact, transformation_artifact= transformation_artifact)
            evaluation.initiate_model_evaluation()
        except Exception as e:
            raise e
        
    def ModelPusherPipeline(self):
        try:
            PusherConfig = self.config.get_model_pusher_config()
            Pushing = ModelPusher(model_pusher_config=PusherConfig)
            Pushing.initiate_model_pusher()
        except Exception as e:
            raise e




    def initiatePipeline(self):
        
        STAGE_NAME = "Data Ingestion"
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
        self.dataingestionpipeline()
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")
        STAGE_NAME = "Prepare Base Model"
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
        self.basemodelpipeline()
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")
        STAGE_NAME = "Data Transformation"
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
        artifact = self.datatransformationpipeline()
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")
        STAGE_NAME = "Model training"
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
        training_artifact = self.trainingpipeline(artifact=artifact)
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")
        STAGE_NAME = "Model Evaluation"
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
        self.evaluationpipeline(transformation_artifact=artifact, training_artifact=training_artifact)
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")
        STAGE_NAME = "Model Pusher"
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
        self.ModelPusherPipeline()
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")




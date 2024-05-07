from Food_Classification.components.data_transformation import data_transformation
from Food_Classification.components.tensorboard import preparetensorboard
from Food_Classification.components.prepare_base_model import PrepareBaseModel
from Food_Classification.config.configuration import ConfigurationManager
from Food_Classification.components.model_trainier import Model_training
from Food_Classification.components.data_ingestion import DataIngestion
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
            transformation_config = self.config.get_data_transform_config()
            transform = data_transformation(config=transformation_config)
            data_transformation_artifact = transform.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise e
        

    def trainingpipeline(self,artifact):
        try:
            tensorboard = preparetensorboard(config=self.config.get_tensorboard_config())
            writer = tensorboard.get_summary_writer()
            train_config = self.config.get_training_config()
            training = Model_training(train_config, artifact=artifact, loss_function= torch.nn.CrossEntropyLoss(), optimizer= torch.optim.Adam,writer=writer)
            training.initiate_training()
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
        self.trainingpipeline(artifact=artifact)
        logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")



if __name__ == "__main__":
    training = TrainingPipeline(ConfigurationManager)
    training.initiatePipeline()
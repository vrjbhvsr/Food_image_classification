from Food_Classification.config.configuration import ConfigurationManager
#from Food_Classification.components.prepare_base_model import PrepareBaseModel
from Food_Classification import logger

STAGE_NAME = "Prepare Base Model"

class PreapareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage : {STAGE_NAME} <<<<<<<<")
        Preapare_base_model = PreapareBaseModelTrainingPipeline()
        Preapare_base_model.main()
        logger.info(f">>>>>>> stage : {STAGE_NAME} completed <<<<<<<< \n\nx========x")

    except Exception as e:
        logger.exception(e)
        raise e

from Food_Classification.config.configuration import ConfigurationManager
from Food_Classification.components.data_ingestion import DataIngestion
from Food_Classification import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config = data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.unzip_file()
        except Exception as e:
            raise e
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage : {STAGE_NAME} <<<<<<<<")
        Data_Ingestion = DataIngestionTrainingPipeline()
        Data_Ingestion.main()
        logger.info(f">>>>>>> stage : {STAGE_NAME} completed <<<<<<<< \n\n")

    except Exception as e:
        logger.exception(e)
        raise e

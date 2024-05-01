from Food_Classification import logger
from Food_Classification.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f"\n>>>>>>>>>>>> stage : {STAGE_NAME} <<<<<<<<<<<<")
    Data_ingestion = DataIngestionTrainingPipeline()
    Data_ingestion.main()
    logger.info(f"\n>>>>>>>>>>>> stage : {STAGE_NAME} completed <<<<<<<<<<<<\n\n ")

except Exception as e:
    logger.exception(e)
    raise e

from Food_Classification import logger
from Food_Classification.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from Food_Classification.pipeline.stage_2_prepare_base_model import PreapareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
    Data_ingestion = DataIngestionTrainingPipeline()
    Data_ingestion.main()
    logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} completed \n ===========================================================================================\n\n ")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
    Preapare_base_model = PreapareBaseModelTrainingPipeline()
    Preapare_base_model.main()
    logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} Completed\n ===========================================================================================\n\n ")

except Exception as e:
    logger.exception(e)
    raise e
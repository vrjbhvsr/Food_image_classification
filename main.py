from Food_Classification import logger
from Food_Classification.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from Food_Classification.pipeline.stage_2_prepare_base_model import PreapareBaseModelTrainingPipeline
from Food_Classification.pipeline.stage_3_data_transformation_pipeline import DataTransformationPipeline
from Food_Classification.pipeline.stage_4_Model_training_pipeline import ModelTrainingPipeline

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

STAGE_NAME = "data Transformation"

try:
    logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME}\n ===========================================================================================\n\n ")
    data_transformations = DataTransformationPipeline()
    data_transformation_artifact = data_transformations.main()
    logger.info(f"\n\n================================================================================\n stage : {STAGE_NAME} Completed\n ===========================================================================================\n\n ")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"

try:
        logger.info(f">>>>>>> stage : {STAGE_NAME} <<<<<<<<")
        Model_train = ModelTrainingPipeline()
        Model_train.main(artifact=data_transformation_artifact)
        logger.info(f">>>>>>> stage : {STAGE_NAME} completed <<<<<<<< \n\nx========x")

except Exception as e:
        logger.exception(e)
        raise e
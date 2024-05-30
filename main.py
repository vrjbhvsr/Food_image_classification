from Food_Classification import logger
from Food_Classification.pipeline.Training_pipeline import TrainingPipeline
from Food_Classification.config.configuration import ConfigurationManager


def start_training():
    try:
        training = TrainingPipeline(ConfigurationManager)
        training.initiatePipeline()
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    start_training()
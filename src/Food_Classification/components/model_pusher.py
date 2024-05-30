import os
import sys
from Food_Classification import logger
from Food_Classification.entity.config_entity import ModelPusherConfig
from Food_Classification.entity.artifact_entity import modelPusherArtifact

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.config = model_pusher_config

    def build_push_bento_image(self):
        logger.info("Entered build_and_push_bento_image method of ModelPusher class")
        try:
            logger.info("Building the bento from bentofile.yaml")

            os.system("bentoml build")

            logger.info("Built the bento from bentofile.yaml")

            """logger.info("Creating docker image for bento")

            os.system(
                f"bentoml containerize {self.config.bentoml_service_name}:latest -t public.ecr.aws/j7a2g6e7/food_classification-094{self.config.bentoml_ecr_image}:latest"
            )

            logger.info("Created docker image for bento")

            logger.info("logger into ECR")

            os.system(
                "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/j7a2g6e7/food_classification-094"
            )

            logger.info("Logged into ECR")

            logger.info("Pushing bento image to ECR")

            os.system(
                f"docker push 637423233018.dkr.ecr.eu-north-1.amazonaws.com/{self.config.bentoml_ecr_image}:latest"
            )

            logger.info("Pushed bento image to ECR")

            logger.info(
                "Exited build_and_push_bento_image method of ModelPusher class"
            )"""

        except Exception as e:
            raise e
    

    def initiate_model_pusher(self) -> modelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """
        logger.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            self.build_push_bento_image()

            model_pusher_artifact = modelPusherArtifact(
                bentoml_model_name=self.config.bentoml_model_name,
                bentoml_service_name=self.config.bentoml_service_name,
            )

            logger.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise e
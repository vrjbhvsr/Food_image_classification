import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from Food_Classification import logger
from Food_Classification.entity.artifact_entity  import DataTransformationArtifact, ModelTrainingArtifact, ModelEvaluationArtifact
from Food_Classification.entity.config_entity import EvaluationConfig



class Model_Evaluation:
    def __init__(self,
                 config: EvaluationConfig, transformation_artifact: DataTransformationArtifact,
                 training_artifact: ModelTrainingArtifact):
        self.config = config
        self.data_transformation_artifact: DataTransformationArtifact = transformation_artifact
        self.model_training_artifact: ModelTrainingArtifact = training_artifact
        
    def test_model(self) -> float:
        '''To test model
        input: model, device, test_dataloader, loss_function
        output: loss, accuracy'''

        test_dataloader: DataLoader = self.data_transformation_artifact.transformed_test_object
        model: Module = torch.load(self.model_training_artifact.trained_model_path)
        model.to(self.config.device)
        loss_function = self.config.loss_function
        model.eval()
        test_loss, test_accuracy = 0, 0

        with torch.inference_mode():
            for batch, (data, label) in enumerate(test_dataloader):
                data, label = data.to(self.config.device), label.to(self.config.device)
                label_pred = model(data)
                loss = loss_function(label_pred, label)
                test_loss += loss.item()

                # Calculate the accuracy
                predictions = torch.argmax(label_pred, dim=1)
                test_accuracy += (predictions == label).sum().item()

        test_loss = test_loss / len(test_dataloader)
        test_accuracy = test_accuracy / len(test_dataloader.dataset)

        logger.info(f"Test loss: {test_loss}")
        logger.info(f"Test accuracy: {test_accuracy}")

        return test_accuracy

        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Model evaluation started")
            
            Accuracy = self.test_model()
            
            logger.info(f"Accuracy: {Accuracy}")
           
            model_evaluation_artifact: ModelEvaluationArtifact = ModelEvaluationArtifact(Model_accuracy= Accuracy)
            
            logger.info("Model evaluation finished")
            
            return model_evaluation_artifact
        except Exception as e:
            raise e




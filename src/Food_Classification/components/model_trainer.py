import torch
import joblib
import bentoml
import os
from pathlib import Path
from Food_Classification.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact
from Food_Classification.entity.config_entity import TrainingConfig
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, _LRScheduler, ReduceLROnPlateau
from Food_Classification import logger
from Food_Classification.config.configuration import ConfigurationManager


class Model_Training:
    def __init__(self, config: TrainingConfig, transformation_Artifacts: DataTransformationArtifact):
        self.config = config
        self.transformation_Artifacts: DataTransformationArtifact = (transformation_Artifacts)
        self.model = torch.load(Path("artifacts/prepare_base_model/base_model_updated.pth"),map_location=torch.device(self.config.device))

    def train_step(self,optimizer: torch.optim.Optimizer):
        '''To train model
        input: model, device, train_dataloader, optimizer, epoch
        output: loss, accuracy'''
        logger.info("Model training train step started")
        try:
            self.model.train()

            train_loss, train_accuracy =0,0
            ProgressBar = tqdm(self.transformation_Artifacts.transformed_train_object)


            for batch, (data, label) in enumerate(ProgressBar):
                data, label = data.to(self.config.device), label.to(self.config.device)

                #forward pass
                y_pred = self.model(data)

                # Calculate the loss 
                loss = self.config.loss_function(y_pred, label)
                train_loss += loss.item()

                #setting optimizer to zero grad
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Calculate the accuracy
                target_predict = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
                train_accuracy += (target_predict == label).sum().item()/len(y_pred)

            train_loss = train_loss/len(self.transformation_Artifacts.transformed_train_object)
            train_accuracy = train_accuracy/len(self.transformation_Artifacts.transformed_train_object)
            logger.info(f"Train_loss: {train_loss} | Train_accuracy: {train_accuracy}|")
            logger.info("Train step is finished")
              
        except Exception as e:
            raise e

    def test_step(self):
        '''To test model
        input: model, device, test_dataloader
        output: loss, accuracy'''
        logger.info("Model training test step started")

        try:
            self.model.eval()

            test_loss, test_accuracy =0,0
            ProgressBar = tqdm(self.transformation_Artifacts.transformed_test_object)
            with torch.inference_mode():
            
                for batch, (data, label) in enumerate(ProgressBar):
                    data, label = data.to(self.config.device), label.to(self.config.device)

                    label_pred = self.model(data)

                    #calcuate the loss
                    loss = self.config.loss_function(label_pred, label)
                    test_loss += loss.item()

                    # Calculate the accuracy
                    target_predict = torch.argmax(torch.softmax(label_pred,dim=1),dim=1)
                    test_accuracy += (target_predict == label).sum().item()/len(label_pred)

            test_loss = test_loss/len(self.transformation_Artifacts.transformed_test_object)
            test_accuracy = test_accuracy/len(self.transformation_Artifacts.transformed_test_object)

            logger.info(f"Test_loss: {test_loss} | Test_accuracy: {test_accuracy}|")
            logger.info("Test step finished")

            return test_loss,test_accuracy
        except Exception as e:
            raise e


    def initiate_Model_training(self) -> ModelTrainingArtifact:
        try:
            logger.info("Model training started")
            model: torch.nn.Module = self.model.to(self.config.device)
            optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(),lr = self.config.learning_rate, weight_decay=1e-5)
            #schedular: _LRScheduler = StepLR(optimizer=optimizer,
            #                                 **self.config.schedular_params)

            schedular: _LRScheduler = ReduceLROnPlateau(optimizer=optimizer, **self.config.schedular_params)
            best_test_loss = float('inf')
            patience = 5
            epochs_no_improve = 0
            early_stop = False
            
            for epoch in range(1,self.config.epochs+1):
                logger.info(f"Epoch: {epoch}")
                self.train_step(optimizer=optimizer)
                optimizer.step()
                test_loss,test_accuracy = self.test_step()
                schedular.step(test_accuracy)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    epochs_no_improve = 0
                    best_model = model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info("Early Stopping")
                    early_stop = True
                    break

            if early_stop:
                model.load_state_dict(best_model)

            trained_model_path = os.path.join(self.config.root_dir, "trained_model.pth")
            torch.save(model, trained_model_path)

            train_trans_obj  = joblib.load(self.transformation_Artifacts.trained_transformed_file)
            bentoml.pytorch.save_model(name= self.config.bentoml_model_name,
                                       model=model,
                                       custom_objects={self.config.train_transform_key: train_trans_obj}
                                       )
            
            
            model_training_artifact: ModelTrainingArtifact = ModelTrainingArtifact(
                trained_model_path= trained_model_path)
            
            logger.info("MOdel training Complete")
            return model_training_artifact

        except Exception as e:
            raise e

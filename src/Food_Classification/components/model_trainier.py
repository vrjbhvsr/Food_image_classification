from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from Food_Classification import logger
from Food_Classification.config.configuration import training_config
from Food_Classification.entity.artifact_entity import DataTransformationArtifact



class Model_training:
    def __init__(self, config: training_config, 
                        artifact: DataTransformationArtifact,
                        loss_function: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer, 
                        writer: SummaryWriter):
        self.config = config
        self.artifact: DataTransformationArtifact = artifact
        self.train_dataloader = self.artifact[0]
        self.test_dataloader = self.artifact[1]
        self.model = torch.load(self.config.model_path,map_location=torch.device(self.config.device))
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.writer = writer
    
    def train_step(self):
        model = self.model
        optimizer = self.optimizer(params= self.model.parameters(),lr = self.config.learning_rate)
        try:
            # Initiate model training
            model.train()
            train_loss = 0
            train_acc= 0
            progress = tqdm(self.train_dataloader)
            # Looping through batches
            for batch, (data, target) in enumerate(progress):
                # Setting up data into device
                data, target = data.to(self.config.device), target.to(self.config.device)

                # forward propogation
                target_pred = model(data)

                # Calculating loss
                loss = self.loss_function(target_pred, target)
                train_loss += loss.item()
                
                # setting optimizer to zero gradient
                optimizer.zero_grad()
                
                # backward propogation
                loss.backward()
                
                # updating weights
                optimizer.step()


                
                # Calculating accuracy
                target_pred_class = torch.argmax(torch.softmax(target_pred, dim=1), dim=1) 
                train_acc += (target_pred_class == target).sum().item() / len(target_pred)

            train_loss /= len(self.train_dataloader)
            train_acc /= len(self.train_dataloader)
            progress.set_description(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            return train_loss, train_acc
                
            
            
        except Exception as e:
            raise e


    def test_step(self):
        model = self.model
        try:
            # Intiatig model evaluation
            model.eval()

            test_loss, test_acc = 0,0
            progress = tqdm(self.test_dataloader)

            with torch.inference_mode():
                for batch, (data, target) in enumerate(progress):
                    data, target = data.to(self.config.device), target.to(self.config.device)

                    # forward propogation
                    target_pred = model(data)

                    # Calculating Loss
                    loss = self.loss_function(target_pred, target)
                    test_loss += loss.item()

                    # calculating Accuracy
                    target_pred_labels = target_pred.argmax(dim=1)
                    test_acc += ((target_pred_labels) == target).sum().item()/len(target_pred_labels)

            test_loss = test_loss/len(self.test_dataloader)
            test_acc = test_acc /len(self.test_dataloader)
            return test_loss, test_acc
        except Exception as e:
            raise e


    def initiate_training(self):
        model = self.config.model_path
        results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
        }

        for epoch in tqdm(range(self.config.epochs)):
            train_loss, train_acc = self.train_step()
            test_loss, test_acc = self.test_step()

            logger.info(f"Epoch: {epoch+1} | "
                f"self.train_loss: {train_loss:.4f} | "
                f"self.train_acc: {train_acc:.4f} | "
                f"self.test_loss: {test_loss:.4f} | "
                f"self.test_acc: {test_acc:.4f}")

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # Save the model
        trained_model = os.path.join(self.config.root_dir, "trained_model.pth")
        torch.save(model,trained_model)

         ### Experiment tracking
        self.writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict = {"Train_loss" : train_loss,
                                                "Test_loss" : test_loss},
                            global_step= epoch
                            )
        self.writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict = {"Train_acc" : train_acc,
                                                "Test_acc" : test_acc},
                            global_step= epoch
                            )

       # self.writer.add_graph(model=model,
       # input_to_model = torch.randn(32,3,224,224).to(self.config.device))
    # Return the filled results at the end of the epochs
        return results

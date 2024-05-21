import torch
import torchvision
from torchinfo import summary
from Food_Classification.config.configuration import PrepareBasemodelConfig 
from pathlib import Path
from Food_Classification import logger
from torch import nn

class PrepareBaseModel:
    def __init__(self,config: PrepareBasemodelConfig):
        self.config = config

    def get_base_model(self):
        self.base_model = torchvision.models.resnet101(weights= self.config.params_weight).to(self.config.params_device)
        torch.save(self.base_model, self.config.base_model_dir)

    @staticmethod
    def preparebasemodel(model, classes, freeze:bool,dropout_rate):
        logger.info(f"\n\n********************************{model.__class__.__name__}*********************************\n\n")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            logger.info(f"\n\n********************************Unfreezing 4th layer parameters*********************************\n\n")
            for param in model.layer4.parameters():
                param.requires_grad = True

            for param in model.fc.parameters():
                param.requires_grad = True

        if model.__class__.__name__ == "EfficientNet":
            model = model
            model.classifier = torch.nn.Sequential(
                        nn.Dropout(p=dropout_rate, inplace=True),
                        nn.Linear(in_features=model.classifier[1].in_features, out_features=classes, bias=True)
                            )               
        else:
            model = model
            model.fc = torch.nn.Sequential(torch.nn.Dropout(p=dropout_rate),
                                           torch.nn.Linear(in_features=model.fc.in_features, out_features= classes))


        info = summary(model= model,input_size=(1,3,224,224),
        col_names=['input_size', 'output_size', 'num_params', "trainable"],
        col_width=20,
        row_settings=["var_names"])
        logger.info(info)
        return model

    def update_base_model(self):
        self.model = self.preparebasemodel(model= self.base_model, 
        classes= self.config.params_classes,
        freeze = True,
        dropout_rate=self.config.p)
        torch.save(self.model, self.config.updated_base_model)

    @staticmethod
    def save_model(path:Path, model: torch.nn.Module):
        torch.save(model, path)

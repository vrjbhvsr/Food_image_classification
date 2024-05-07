import torch
import torchvision
from torchinfo import summary
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBasemodelConfig):
        self.config = config

    def get_base_model(self):
        self.base_model = torchvision.models.efficientnet_b4(weights=self.config.params_weight).to(self.config.params_device)
        torch.save(self.base_model.state_dict(), self.config.base_model_dir)

    @staticmethod
    def prepare_base_model(model, classes, freeze: bool):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=model.classifier[0].p),
            torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=classes)
        )

        info = summary(model=model, input_size=(1, 3, 224, 224),
                       col_names=['input_size', 'output_size', 'num_params', "trainable"],
                       col_width=20,
                       row_settings=["var_names"])
        return model, info

    def update_base_model(self):
        self.model, _ = self.prepare_base_model(model=self.base_model, 
                                                classes=self.config.params_classes,
                                                freeze=True)
        torch.save(self.model.state_dict(), self.config.updated_base_model)

    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        torch.save(model.state_dict(), path)

from Food_Classification.config.configuration import ConfigurationManager
from Food_Classification.entity.config_entity import DataTransformConfig
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from Food_Classification.entity.artifact_entity import DataTransformationArtifact
from Food_Classification import logger
from typing import Tuple
import joblib
import os

class DataTransformation:
    def __init__(self,config: DataTransformConfig):
        self.config = config

    def train_data_transform(self):
        try:
            logger.info("Data Transformation started")
            train_transforms: transforms.Compose = transforms.Compose([transforms.Resize(self.config.spatial_transform['resize']),
                                                                      transforms.CenterCrop(self.config.spatial_transform['center_crop']),
                                                                      transforms.RandomRotation(self.config.spatial_transform['rotation']),
                                                                      transforms.RandomVerticalFlip(self.config.spatial_transform['vertical_flip']),
                                                                      transforms.ColorJitter(**self.config.color_transform),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(**self.config.normalize)])

        
            return train_transforms
        except Exception as e:
            raise e
    
    def test_data_transform(self):
        try:
            test_transforms: transforms.Compose = transforms.Compose([transforms.Resize(self.config.spatial_transform['resize']),
                                                                    transforms.CenterCrop(self.config.spatial_transform['center_crop']),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(**self.config.normalize)])

            return test_transforms
        except Exception as e:
            raise e
    

    def create_dataloaders(self, train_data:transforms.Compose, test_data: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
        try:
            logger.info("Data Loading started")
            train_dataset = ImageFolder(root= self.config.train_dir,
                                        transform=train_data)
            
            test_dataset = ImageFolder(root=self.config.test_dir,
                                    transform=test_data)
            

            train_dataloader = DataLoader(train_dataset, **self.config.data_loader_params)

            test_dataloader = DataLoader(test_dataset, **self.config.data_loader_params)

            return train_dataloader, test_dataloader
        except Exception as e:
            raise e
    
    def initiate_datatransfrom(self) -> DataTransformationArtifact:
        try:
            train_transforms: transforms.Compose = self.train_data_transform()
            test_transforms: transforms.Compose = self.test_data_transform()

            joblib.dump(train_transforms, self.config.train_transforms_file)
            joblib.dump(test_transforms, self.config.test_transforms_file)
            logger.info("transform pickle file created")

            train_dataloader, test_dataloader = self.create_dataloaders(train_transforms, test_transforms)

            transforms_artifact: DataTransformationArtifact = DataTransformationArtifact(transformed_train_object= train_dataloader,
                                                                                         transformed_test_object= test_dataloader,
                                                                                         trained_transformed_file= self.config.train_transforms_file,
                                                                                         test_transformed_file= self.config.test_transforms_file)
            logger.info("Data Transformation completed")
            return transforms_artifact
        except Exception as e:
            raise e

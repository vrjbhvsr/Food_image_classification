import os
from typing import Tuple
import joblib
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from Food_Classification import logger
from Food_Classification.entity.config_entity import data_transformation_config

class data_transformation:
    def __init__(self, config: data_transformation_config):
        self.config = config

    def tranform_train_data(self) -> transforms.Compose:
        try: 
            logger.info("Transforming train data")
            train_transform: transforms.Compose = transforms.Compose([transforms.Resize(self.config.spatial_transform['resize']),
                                                                      transforms.CenterCrop(self.config.spatial_transform['center_crop']),
                                                                      transforms.RandomRotation(self.config.spatial_transform['rotation']),
                                                                      transforms.RandomVerticalFlip(self.config.spatial_transform['vertical_flip']),
                                                                      transforms.ColorJitter(**self.config.color_transform),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(**self.config.normalize_transform)])
            

            return train_transform
        except Exception as e:
            raise e
        
    def test_transform(self) -> transforms.Compose:
        try:
            logger.info("Transforming test data")
            test_transform: transforms.Compose = transforms.Compose([transforms.Resize(self.config.spatial_transform['resize']),
                                                                      transforms.CenterCrop(self.config.spatial_transform['center_crop']),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(**self.config.normalize_transform)])

            return test_transform
        
        except Exception as e:
            raise e


    def create_dataloaders(self, train_transform: transforms.Compose, test_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
        try:
            logger.info("creating_dataloaders")

            train_data: Dataset = ImageFolder(root =self.config.train_dir, 
                                              transform= train_transform)

            test_data: Dataset = ImageFolder(root= self.config.test_dir, 
                                             transform = test_transform)
            
            class_names: list = train_data.classes

            train_dataloader: DataLoader = DataLoader(dataset= train_data,
                                                      batch_size= self.config.batch_size,
                                                      shuffle = self.config.shuffle,
                                                      **self.config.data_loader_params)
            
            test_dataloader: DataLoader = DataLoader(dataset= test_data,
                                                      batch_size= self.config.batch_size,
                                                      shuffle = False,
                                                      **self.config.data_loader_params)
            
            logger.info('DataLoaders created')
            return train_dataloader, test_dataloader, class_names
        
        except Exception as e:
            raise e
    
        
    def initiate_data_transformation(self):
        try:
            logger.info("Initiating data transformation")

            train_transform: transforms.Compose = self.tranform_train_data()

            test_transform: transforms.Compose = self.test_transform()

            train_transform_filename = os.path.join(self.config.transforms_pkl, "train_transforms.pkl")  # Create filename with path and extension
            test_transform_filename = os.path.join(self.config.transforms_pkl, "test_transforms.pkl")

            joblib.dump(train_transform, train_transform_filename)
            joblib.dump(test_transform, test_transform_filename)


            train_dataloader, test_dataloader, class_name = self.create_dataloaders(train_transform=train_transform, test_transform=test_transform)
            
            self.config.transformed_train_data = train_dataloader
            self.config.transformed_test_data = test_dataloader

            return self.config.transformed_train_data, self.config.transformed_test_data, class_name
        
        except Exception as e:
            raise e

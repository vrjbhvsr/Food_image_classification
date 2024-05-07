from Food_Classification.constants import *
from Food_Classification.utils.common import read_yaml,create_directory
from Food_Classification.entity.config_entity import DataIngestionConfig, PrepareBasemodelConfig, preparetensorboardconfig, data_transformation_config, training_config
class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

        create_directory([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion

            create_directory([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir= config.root_dir,
                source_URL= config.source_url,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir)
            
            return data_ingestion_config

    def get_base_model_config(self) -> PrepareBasemodelConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directory([config.root_dir])

        prepare_base_model_cofig = PrepareBasemodelConfig(
                                                        root_dir= Path(config.root_dir),
                                                        base_model_dir= Path(config.base_model_path),
                                                        updated_base_model= Path(config.updated_base_model),
                                                        params_image_size=  params.IMAGE_SIZE,
                                                        params_device= params.DEVICE,
                                                        params_weight= params.WEIGHTS,
                                                        params_classes= params.CLASSES
                                                        )
        
        return prepare_base_model_cofig

    

    def get_tensorboard_config(self) -> preparetensorboardconfig:

            config = self.config.prepare_tensorboard
            create_directory([config.root_dir])
            create_directory([config.tensorboard_root_log_dir])

            prepare_tensorboard_config = preparetensorboardconfig(root_dir = config.root_dir,
                                                            tensorboard_root_log_dir = config.tensorboard_root_log_dir,
                                                            experiment_name = self.params.EXPERIMENT_NAME,
                                                            model_name = self.params.MODEL_NAME,
                                                            epochs = self.params.EPOCHS
                                                            )
            
            return prepare_tensorboard_config
        


    def get_data_transform_config(self) -> data_transformation_config:
        config = self.config.data_transforms

        train_dir = os.path.join(self.config.data_ingestion.unzip_dir,'food_40_percent','train')
        test_dir = os.path.join(self.config.data_ingestion.unzip_dir,'food_40_percent','test')
        create_directory([config.root_dir])
        data_transformation_configuration = data_transformation_config(root_dir= config.root_dir,
                                                        train_dir= Path(train_dir),
                                                        test_dir= Path(test_dir),
                                                        batch_size= self.params.BATCH_SIZE,
                                                        shuffle= self.params.SHUFFLE,
                                                        color_transform ={'brightness': BRIGHTNESS,
                                                                                'contrast': CONTRAST,
                                                                                'saturation': SATURATION,
                                                                                'hue': HUE},
                                                        spatial_transform= {'vertical_flip': VERTICLE_FLIP,
                                                                            'resize': RESIZE,
                                                                            'center_crop': CENTER_CROP,
                                                                            'rotation': RANDOMROTATION
                                                                            },
                                                        normalize_transform= {'mean': NORMALIZE_MEAN,
                                                                                'std': NORMALIZE_STD},
                                                        data_loader_params= {"num_workers": NUM_WORKERS,
                                                                                "pin_memory": PIN_MEMORY})
        
        return data_transformation_configuration


    def get_training_config(self) -> training_config:
        config = self.config.model_training
        create_directory([config.root_dir])
        self.model_path = self.config.prepare_base_model.updated_base_model
        Training_Config = training_config(root_dir= config.root_dir,
                                            model_path= self.model_path,
                                            epochs= self.params.EPOCHS,
                                            batch_size= self.params.BATCH_SIZE,
                                            learning_rate= self.params.LEARNING_RATE,
                                            device=self.params.DEVICE
                                            )


        return Training_Config


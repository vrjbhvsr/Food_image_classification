from Food_Classification.constants import *
from Food_Classification.utils.common import read_yaml,create_directory
from Food_Classification.entity.config_entity import DataIngestionConfig, PrepareBasemodelConfig, preparetensorboardconfig

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
                                                        params_classes= params.CLASSES)
        
        return prepare_base_model_cofig
    

    def get_tensorboard_config(self) -> preparetensorboardconfig:

            config = self.config.prepare_tensorboard
            create_directory([config.tensorboard_root_log_dir])

            prepare_tensorboard_config = preparetensorboardconfig(root_dir = config.root_dir,
                                                            tensorboard_root_log_dir = config.tensorboard_root_log_dir,
                                                            experiment_name = self.params.EXPERIMENT_NAME,
                                                            model_name = self.params.MODEL_NAME,
                                                            epochs = self.params.EPOCHS
                                                            )
            
            return prepare_tensorboard_config
        

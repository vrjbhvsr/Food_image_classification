from Food_Classification.constants import *
from Food_Classification.utils.common import read_yaml,create_directory
from Food_Classification.entity.config_entity import DataIngestionConfig

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
        

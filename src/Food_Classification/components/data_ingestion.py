import os 
from pathlib import Path
import zipfile
from Food_Classification import logger
from Food_Classification.utils.common import get_size
from Food_Classification.entity.config_entity import DataIngestionConfig
import gdown


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config=config
        
    def download_file(self):
        data_url = self.config.source_URL
        if not os.path.exists(self.config.local_data_file):
            file_id = data_url.split('/')[-2]
            prefix = "https://drive.google.com/uc?export=download&id="
            gdown.download(prefix + file_id, self.config.local_data_file)
            logger.info(f"Downloaded {self.config.local_data_file}")

        else:
            logger.info(f"{self.config.local_data_file} already exists of size : {get_size(Path(self.config.local_data_file))}")

    
    def unzip_file(self):
        """
        Unzips the downloaded file.
        Extracts all the files from the zip file into the unzip directory.
        """

        unzip_dir = self.config.unzip_dir
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

            

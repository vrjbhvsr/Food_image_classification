from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from Food_Classification.entity.config_entity import preparetensorboardconfig

class preparetensorboard:
    def __init__(self, config: preparetensorboardconfig):
        self.config = config

    def create_summary_writer(self,experiment_name, model_name, extra: str = None) -> SummaryWriter:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M")

        if extra:
            log_dir = os.path.join(self.config.tensorboard_root_log_dir, timestamp, experiment_name, model_name,extra)
        else:
            log_dir = os.path.join(self.config.tensorboard_root_log_dir, timestamp, experiment_name, model_name)

        return SummaryWriter(log_dir= log_dir)
    
    def get_summary_writer(self):
        self.summary = self.create_summary_writer(experiment_name=self.config.experiment_name,
                                                  model_name= self.config.model_name,
                                                  extra= f"{self.config.epochs} epochs")
        
        return self.summary

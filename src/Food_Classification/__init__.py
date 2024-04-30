import os
import sys
import logging as log

log_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir,exist_ok=True)

log.basicConfig(
    level = log.INFO,
    format=log_str,

    handlers=[
        log.FileHandler(log_filepath),
        log.StreamHandler(sys.stdout)
    ]
)

logger = log.getLogger('Food_Classification_logger')
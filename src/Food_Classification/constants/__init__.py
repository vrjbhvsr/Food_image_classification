from pathlib import Path
from typing import List
import os
import torch

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")


BRIGHTNESS: float = 0.10
CONTRAST:float = 0.10
SATURATION: float = 0.2
HUE:float = 0.2
RESIZE:int= 224
CENTER_CROP:int = 224
RANDOMROTATION:int=  10
RANDOMZOOM:float = 0.2
VERTICLE_FLIP:bool = True
NORMALIZE_MEAN: List[int] = [0.485, 0.456, 0.406]
NORMALIZE_STD:List[int] = [0.229, 0.224, 0.225]
SHUFFLE:bool = True
PIN_MEMORY:bool = True
NUM_WORKERS:int = os.cpu_count()

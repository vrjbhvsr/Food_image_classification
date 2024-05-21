from pathlib import Path
from typing import List,Tuple
import os
from torch import nn

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")


BRIGHTNESS: float = 0.20
CONTRAST:float = 0.20
SATURATION: float = 0.2
HUE:float = 0.1
RESIZE:int= 224
CENTER_CROP:int = 224
RANDOMROTATION:int= 10
RANDOMZOOM:float = 0.2
VERTICLE_FLIP:bool = True
NORMALIZE_MEAN: List[int] = [0.485, 0.456, 0.406]
NORMALIZE_STD:List[int] = [0.229, 0.224, 0.225]
KERNEL_SIZE: int = 3
SIGMA: Tuple[float] = (0.4,0.4)
NUM_OUTPUT_CHANNELS: int = 1
SHUFFLE:bool = True
PIN_MEMORY:bool = True
NUM_WORKERS:int = os.cpu_count()
LOSS_FUNCTION = nn.CrossEntropyLoss
DEGREES: int = 20
TRANSLATE: Tuple[float] = (0.1,0.2)
SCALE: Tuple[float] = (0.8,1.2)
SHEAR: int = 20



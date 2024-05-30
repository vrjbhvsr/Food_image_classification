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
DEVICE: str = 'cuda'

TRAIN_TRANSFORM_KEY: str = "TRAIN_TRANSFORM_KEY"

BENTOML_MODEL_NAME: str = 'food_classification_model'

BENTOML_SERVICE_NAME: str = 'food_classification_service'

BENTOML_ECR_URI: str = 'food_classification-097'

PREDICTION_LABEL: dict = {0: 'beet_salad',
 1: 'bruschetta',
 2: 'cheesecake',3: 'chocolate_cake',4: 'fried_rice',
 5: 'frozen_yogurt',
 6: 'garlic_bread',
 7: 'gnocchi',
 8: 'grilled_cheese_sandwich',
 9: 'macaroni_and_cheese',
 10: 'nachos',
 11: 'pancakes',
 12: 'ravioli',
 13: 'samosa',
 14: 'spring_rolls',
 15: 'strawberry_shortcake',
 16: 'tacos',
 17: 'takoyaki',
 18: 'tiramisu',
 19: 'waffles'}

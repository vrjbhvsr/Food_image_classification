from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBasemodelConfig:
    root_dir: Path
    base_model_dir: Path
    updated_base_model: Path
    params_image_size: list
    params_device: str
    params_weight: str
    params_classes: int
    learning_rate: float


@dataclass(frozen =True)
class preparetensorboardconfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    experiment_name: str
    model_name:str
    epochs: int



@dataclass
class data_transformation_config:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    batch_size: int
    shuffle: bool
    color_transform: dict
    spatial_transform: dict
    normalize_transform: dict
    data_loader_params: dict



@dataclass
class training_config:
    root_dir: Path
    model_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
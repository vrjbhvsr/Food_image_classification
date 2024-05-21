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
    p: float
    


@dataclass(frozen =True)
class preparetensorboardconfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    experiment_name: str
    model_name:str
    epochs: int
    



@dataclass
class DataTransformConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    color_transform: dict
    spatial_transform: dict
    data_loader_params: dict
    normalize: dict
    blur: dict
    affine: dict
    train_transforms_file: Path
    test_transforms_file: Path


@dataclass
class TrainingConfig:
    root_dir: Path
    loss_function: torch.nn.Module
    learning_rate: float
    epochs: int
    device: str
    classes: int
    schedular_params: dict
    bentoml_model_name: str
    train_transform_key: str
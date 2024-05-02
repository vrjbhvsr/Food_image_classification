from dataclasses import dataclass
from pathlib import Path

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

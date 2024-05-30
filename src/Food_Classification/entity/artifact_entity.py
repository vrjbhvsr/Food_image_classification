from dataclasses import dataclass
from torch.utils.data import DataLoader
from pathlib import Path
@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader
    transformed_test_object: DataLoader
    trained_transformed_file: str
    test_transformed_file: str


@dataclass
class ModelTrainingArtifact:
    trained_model_path: str

@dataclass
class ModelEvaluationArtifact:
    Model_accuracy: float

@dataclass
class modelPusherArtifact:
    bentoml_model_name: str
    bentoml_service_name: str
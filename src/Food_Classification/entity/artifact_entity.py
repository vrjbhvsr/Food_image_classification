from dataclasses import dataclass
from torch.utils.data import DataLoader
@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader
    transformed_test_object: DataLoader
    
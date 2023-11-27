from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig():
    root_dir: Path
    train_source_URL: str
    test_source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig():
    root_dir: Path
    STATUS_FILE: Path
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig():
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    weight_decay: float
    special_tokens: list

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    classifier_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    onet_embeddings:Path
    lr_model_path:Path
from typing import List

from psutil import cpu_count
from pydantic import BaseModel, ValidationError, validator


class TrainConfigs(BaseModel):
    no_cuda: bool = False
    parallelize_cross_encoder: bool = True
    validation_data_ratio: float = 0.3
    output_dir: str = "outputs"
    batch_size: int = 32
    weight_decay: float = 0.0001
    dataloader_num_workers: int = 2
    model_type: str = "bert"
    negative_labels_per_sample: int = 3  # For parralelized cross encoder only
    early_stopping_epochs: int = 6
    warmup_epochs: float = 1
    encoder_hidden_layers: int = 2
    decoder_hidden_layers: int = 1  # For bart model only
    epochs: int = 10
    validation_threshold: float = 0.5
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    logging_steps: int = 50
    max_length: int = 256

    @validator("model_type")
    def validate_model_type(cls, v):
        if v not in ["bert", "bart"]:
            raise ValidationError('Value must be one of ["bert", "bart"]')
        return v

    @validator("dataloader_num_workers")
    def validate_dataloader_num_workers(cls, v):
        if v is None or v == -1:
            return cpu_count()
        return v


class TestConfigs(BaseModel):
    device: str = "cpu"
    model_path: str
    batch_size: int = 16
    num_labels_per_sample: int = 4  # For parallelized cross-encoders only
    dataloader_num_workers: int = 2

    @validator("dataloader_num_workers")
    def validate_dataloader_num_workers(cls, v):
        if v is None or v == -1:
            return cpu_count()
        return v

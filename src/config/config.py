from dataclasses import dataclass, field
import torch

@dataclass
class DataConfig:
    root_dir: str = 'data'
    val_size: float | None = 0.1
    test_size: float = 0.15
    resize: tuple[int, int] = (128, 128)
    batch_size: int = 32
    grayscale: bool = False

@dataclass
class ModelConfig:
    out_channels: tuple[int, ...] = (31, 64, 128)
    out_dense_hidden_layers: tuple[int, ...] = (512, 256, 64)
    conv_kernel_size: int = 3 
    pool_kernel_size: int = 2

@dataclass
class TrainingConfig:
    num_epochs: int = 30
    lr: float = 0.001
    class_weights: list[float] = field(
            default_factory=lambda: [0.0602, 0.0458, 0.1705, 0.5041, 0.0561, 0.0988, 0.0644]
    )
    scheduler_patience: int = 4 

@dataclass
class Config:
    random_state = 2137
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)


from dataclasses import dataclass

@dataclass
class DataConfig:
    root_dir: str = 'data'
    val_size: float | None = 0.1
    test_size: float = 0.15
    resize: tuple[int, int] = (244,244)
    batch_size: int = 32
    grayscale: bool = True
    random_state: int = 2137

@dataclass
class ModelConfig:
    in_channels: int = 1 
    out_channels: tuple[int, ...] = (31, 34, 128)
    out_dense_hidden_layers: tuple[int, ...] = (512, 256, 64)
    conv_kernel_size: int = 3 
    pool_kernel_size: int = 2


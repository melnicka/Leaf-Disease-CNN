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



from dataclasses import dataclass, field

@dataclass
class DataConfig:
    """Data configuation schema.

    Attributes:
        root_dir: The root data directory.
        val_size: Size of the validation set. If None, the validation set will not be created.
        test_size: Size of the test set.
        resize: Shape of transformed images.
        batch_size: Batch size for data loaders.
        grayscale: Whether to convert images to grayscale.
        class_weights: if true, calculate class weights to account for class imbalances.
    """
    root_dir: str = 'data'
    val_size: float | None = 0.1
    test_size: float = 0.15
    resize: tuple[int, int] = (128, 128)
    batch_size: int = 32
    grayscale: bool = False
    num_classes: int = 7
    class_weights: bool = True

@dataclass
class ModelConfig:
    """Model configuration schema.

    Attributes:
        out_channels: Controls the number of convolution blocks and the 
            number of channels produced by each convolution block. 
        dense_hidden_dims: Controls the number of hidden dense layers and the
            number of output neurons in each dense layer.
        conv_kernel_size: Size of the convolving kernel.
        pool_kernel_size: Size of the max pooling kernel.
    """
    out_channels: tuple[int, ...] = (31, 64, 128)
    dense_hidden_dims: tuple[int, ...] = (512, 256, 64)
    conv_kernel_size: int = 3 
    pool_kernel_size: int = 2
    dropout_rate: float = 0.3

@dataclass
class TrainingConfig:
    """Training configuration schema.

    Attributes:
        num_epochs: The number of training epochs.
        lr: Learning rate.
        scheduler_patience: The patience of ReduceLROnPlateau scheduler.
    """
    num_epochs: int = 30
    lr: float = 0.001
    scheduler_patience: int = 5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

@dataclass 
class Config:
    """The complete configuration class.

    Attributes:
        random_state: Seed for the entire training process.
        data: Data config.
        model: Model config.
        train: Training config.
    """
    random_state: int = 2137
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)


import torch
import torch.nn as nn 
from .config_schema import Config

class LeafCNN(nn.Module):
    """A convolutional neural network with architecture based on the configuration object.

    Attributes:
        conv_blocks: Convolutional blocks (Conv2d, MaxPool2d, ReLU).
        dense_layers: Dense layers with ReLU activation function.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.conv_blocks = _build_conv_blocks(cfg)
        self.dense_layers= _build_dense_layers(cfg)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

def _build_conv_blocks(cfg: Config) -> nn.Sequential:
    """Builds convolutional blocks (Conv2d, MaxPool2d, ReLU) based on the config.

    Args:
        cfg: Configuration object.

    Returns:
        nn.Sequential: Convolutional blocks.
    """
    conv_list = []

    in_channels = 1 if cfg.data.grayscale else 3
    for out_channels in cfg.model.out_channels:
        conv_list.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg.model.conv_kernel_size
                )
            )
        conv_list.append(nn.ReLU())
        conv_list.append(nn.MaxPool2d(cfg.model.pool_kernel_size))
        in_channels = out_channels

    return nn.Sequential(*conv_list)

def _build_dense_layers(cfg: Config) -> nn.Sequential:
    """Builds dense layers based on the config.

    Args:
        cfg: Configuration object.

    Returns:
        nn.Sequential: Dense layers.
    """
    dense_list = []
    dense_list.append(nn.LazyLinear(cfg.model.dense_hidden_dims[0]))
    dense_list.append(nn.ReLU())
    dense_list.append(nn.Dropout(cfg.model.dropout_rate))

    in_features = cfg.model.dense_hidden_dims[0]
    for out_features in cfg.model.dense_hidden_dims[1:]:
        dense_list.append(nn.Linear(in_features, out_features))
        dense_list.append(nn.ReLU())
        in_features = out_features

    dense_list.append(nn.Dropout(cfg.model.dropout_rate))
    dense_list.append(nn.Linear(in_features, cfg.data.num_classes))

    return nn.Sequential(*dense_list)



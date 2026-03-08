import torch
import torch.nn as nn 
from .config.config import Config

class LeafCNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.conv_blocks = _build_conv_blocks(cfg)
        self.dense_layers= _build_dense_layers(cfg, num_classes=7)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

def _build_conv_blocks(cfg: Config):
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

def _build_dense_layers(cfg: Config, num_classes: int):
    dense_list = []
    dense_list.append(nn.LazyLinear(cfg.model.out_dense_hidden_layers[0]))
    dense_list.append(nn.ReLU())

    in_features = cfg.model.out_dense_hidden_layers[0]
    for out_features in cfg.model.out_dense_hidden_layers[1:]:
        dense_list.append(nn.Linear(in_features, out_features))
        dense_list.append(nn.ReLU())
        in_features = out_features

    dense_list.append(nn.Linear(in_features, num_classes))

    return nn.Sequential(*dense_list)



import torch
import torch.nn as nn 
from .config.config import ModelConfig

class LeafCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.conv_blocks = _build_conv_blocks(cfg)
        self.dense_layers= _build_dense_layers(cfg, num_classes=7)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

def _build_conv_blocks(cfg: ModelConfig):
    conv_list = []

    in_channel = cfg.in_channels
    for out_channel in cfg.out_channels:
        conv_list.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=cfg.conv_kernel_size
                )
            )
        conv_list.append(nn.ReLU())
        conv_list.append(nn.MaxPool2d(cfg.pool_kernel_size))
        in_channel = out_channel

    return nn.Sequential(*conv_list)

def _build_dense_layers(cfg: ModelConfig, num_classes: int):
    dense_list = []
    dense_list.append(nn.LazyLinear(cfg.out_dense_hidden_layers[0]))
    dense_list.append(nn.ReLU())

    in_features = cfg.out_dense_hidden_layers[0]
    for out_features in cfg.out_dense_hidden_layers[1:]:
        dense_list.append(nn.Linear(in_features, out_features))
        dense_list.append(nn.ReLU())
        in_features = out_features

    dense_list.append(nn.Linear(in_features, num_classes))

    return nn.Sequential(*dense_list)



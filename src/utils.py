from collections import Counter
import numpy as np
import torch
import random
from .model import LeafCNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf
from .config.config import Config

def load_config(yaml_cfg_path: str = "src/config/default.yaml") -> Config:
    schema = OmegaConf.structured(Config)
    yaml_cfg = OmegaConf.load(yaml_cfg_path)
    cfg = OmegaConf.merge(schema, yaml_cfg)
    cfg = OmegaConf.to_object(cfg)

    return cfg

def training_setup(cfg: Config) -> tuple[LeafCNN, AdamW, CrossEntropyLoss, ReduceLROnPlateau]:
    model = LeafCNN(cfg)
    optimizer = AdamW(model.parameters(), cfg.train.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg.train.scheduler_patience)
    class_weights = torch.tensor(cfg.train.class_weights)
    criterion = CrossEntropyLoss(weight=class_weights)

    return model, optimizer, criterion, scheduler

def set_random_state(cfg: Config):
    np.random.seed(cfg.random_state)
    random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    torch.cuda.manual_seed_all(cfg.random_state)

def calculate_class_weights(labels: list[int]) -> list[float]:
    weights = []
    counter = Counter(labels)
    weight_sum = 0.0
    total_samples = sum(counter.values())
    print(sorted(counter))
    for label in sorted(counter):
        weight = total_samples / (7.0 * counter[label])
        weights.append(weight)
        weight_sum += weight

    weights = [float(np.round(w / weight_sum, 4)) for w in weights]

    return weights


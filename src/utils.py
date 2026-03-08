from collections import Counter
import numpy as np
import torch
from .model import LeafCNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

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

def training_setup(cfg: Config):
    model = LeafCNN(cfg)
    optimizer = AdamW(model.parameters(), cfg.train.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg.train.scheduler_patience)
    class_weights = torch.tensor(cfg.train.class_weights)
    criterion = CrossEntropyLoss(weight=class_weights)

    return model, optimizer, criterion, scheduler

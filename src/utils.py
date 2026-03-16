import numpy as np
import torch
import random
import argparse
from collections import Counter
from .model import LeafCNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf
from .config_schema import Config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    pred_parser = subparsers.add_parser("predict")

    train_parser.add_argument("name", help="Experiment name")
    train_parser.add_argument(
            "--config", "-c",
            nargs="+",
            help="YAML config files. Uses base.yaml config by default."
            )
    train_parser.add_argument(
            "--root_dir", "-r",
            default="configs",
            help="Change the directory where config files will be searched."
            )

    pred_parser.add_argument("model_name", help="Name of the trained model.")
    pred_parser.add_argument(
            "input_data",
            nargs="+",
            help="List of files and/or directories with input images."
            )
    pred_parser.add_argument(
            "--root_dir", "-r",
            default="runs",
            help="Change the directory where the model will be searched."
            )

    pred_parser.add_argument(
            "--save", "-s",
            help="The target path to save the predictions."
            )

    args = parser.parse_args()
    return args

def load_config(yaml_cfg_paths, base="configs/base.yaml") -> Config:
    schema = OmegaConf.structured(Config)
    yaml_configs = []
    base_cfg = OmegaConf.load(base)
    if not yaml_cfg_paths:
        cfg = OmegaConf.merge(schema, base_cfg)
        return OmegaConf.to_object(cfg)

    for conf_path in yaml_cfg_paths:
        yaml_configs.append(OmegaConf.load(conf_path))
    cfg = OmegaConf.merge(schema, base_cfg, *yaml_configs)

    return OmegaConf.to_object(cfg)

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


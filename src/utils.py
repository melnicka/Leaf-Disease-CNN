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
    """Parses command-line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
            description=(
                "Detect potato leaf diseases using a convolutional neural network. "
                "The program can train a new model or run predictions with an "
                "existing trained model."
        )   
    )
    subparsers = parser.add_subparsers(
            dest="commands",
            required=True,
            help="Available commands",
    )
    train_parser = subparsers.add_parser(
            "train",
            help="Train a new model and save run artifacts.",
            description=(
            "Train and evaluate a new model. Results are saved in the 'runs/' "
            "directory.\n\n"
            "Model architecture and hyperparameters are defined in YAML "
            "configuration files. Additional configs can be provided to "
            "override the default parameters.\n"
            "Config paths are resolved relative to a configurable base "
            "directory."
        )
            )
    pred_parser = subparsers.add_parser(
        "predict",
        help="Run inference using a trained model.",
        description="Load a trained model and generate predictions for input images.",
    )

    train_parser.add_argument(
        "name",
        help="Experiment name. Also used as the model filename prefix.",
    )

    train_parser.add_argument(
        "--config",
        "-c",
        nargs="+",
        help=(
            "Additional YAML configuration files used to override the "
            "default configuration (base.yaml)."
        ),
    )

    train_parser.add_argument(
        "--base_dir",
        default="configs",
        help="Base directory used to resolve config file paths (default: configs).",
    )

    pred_parser.add_argument(
        "name",
        help="Name of the trained model (experiment name).",
    )

    pred_parser.add_argument(
        "input_data",
        nargs="+",
        help="One or more image files and/or directories containing images.",
    )

    pred_parser.add_argument(
        "--root_dir",
        "-r",
        default="runs",
        help="Directory where trained models are stored (default: runs).",
    )

    pred_parser.add_argument(
        "--save",
        "-s",
        help="Path where prediction results will be saved.",
    )

    args = parser.parse_args()
    return args

def load_config(yaml_cfg_paths: list[str], base: str = "configs/base.yaml") -> Config:
    """Merges user-provided configs files with the base config file and loads it into an object.

    Uses dataclass schema to validate fields in YAML files.

    Args:
        yaml_cfg_paths: Paths to optional config files in YAML format.
        base: Path to the base config file in YAML format.

    Returns:
        A configuration object.
    """
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
    """A helper function to quickly get necessary training objects.

    Args:
        cfg: A configuration object.

    Returns:
        tuple:
        - model: LeafCNN model object.
        - optimizer: AdamW optimizer.
        - criterion: Weighted CrossEntropyLoss function.
        - scheduler: Learning rate scheduler.

    """
    model = LeafCNN(cfg)
    optimizer = AdamW(model.parameters(), cfg.train.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg.train.scheduler_patience)
    class_weights = torch.tensor(cfg.train.class_weights)
    criterion = CrossEntropyLoss(weight=class_weights)

    return model, optimizer, criterion, scheduler

def set_random_state(cfg: Config):
    """Sets up global random seed for better reproducibility.

    Args:
        cfg: A configuration object.
    """
    np.random.seed(cfg.random_state)
    random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    torch.cuda.manual_seed_all(cfg.random_state)

def calculate_class_weights(labels: list[int]) -> list[float]:
    """Calculates the weights for the loss funtion to account for class imbalance.

    Args:
        labels: A list of numeric class labels for all of the samples.

    Returns:
        A list of weights.
    """
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


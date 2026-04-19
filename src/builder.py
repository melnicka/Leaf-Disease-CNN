from pathlib import Path
import torch
import json
from torch.utils.data import DataLoader
from .engine import train, score, predict, DEVICE
from .utils import training_setup, set_random_state, load_config
from .model import LeafCNN
from .dataloading import get_datasets, load_inference_data
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from .config_schema import Config

def train_model(name: str, cfg: Config):
    """Train a model and save all artifacts for the run.

    Creates a directory ``runs/<name>/`` containing:
        - ``<name>.pth`` — model state dictionary
        - ``<name>.yaml`` — configuration used for training
        - ``eval_metrics.json`` — metrics from the final evaluation on the test set

    TensorBoard logs are written to ``runs/tensorboards/<name>/``.

    Args:
        name: Experiment name. Also used as the model filename prefix.
        cfg: Training configuration.
    """
    set_random_state(cfg)
    train_dataset, val_dataset, test_dataset = get_datasets(cfg)
    train_loader = DataLoader(train_dataset, cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, cfg.data.batch_size, shuffle=True)
    
    writer = SummaryWriter(f"runs/tensorboards/{name}")
    model, optimizer, criterion, scheduler = training_setup(cfg, train_dataset)
    model.to(DEVICE)

    train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            scheduler,
            writer
    ) 
    dir_path = Path("runs") / name
    dir_path.mkdir(exist_ok=True, parents=True)
    model_path = dir_path / f"{name}.pth"
    torch.save(model.state_dict(), model_path)
    accuracy, precision, recall, f1 = score(model, test_loader)
    metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
    }
    with open(dir_path / "eval_metrics.json", "w") as f:
        json.dump(metrics, f)

    OmegaConf.save(cfg, dir_path / f"{name}.yaml")
    
def make_predictions(
        model_name: str,
        data_paths: list[str],
        root_dir='runs'
) -> list[str]:
    """Load a trained model and generate predictions for input images.

    The function expects a trained model stored in ``<root_dir>/<model_name>/``.
    The directory must contain:
        - ``<model_name>.pth`` — model state dictionary
        - ``<model_name>.yaml`` — configuration used during training

    The configuration file is loaded to reconstruct the model architecture
    and preprocessing pipeline before running inference.

    Args:
        model_name: Name of the trained model (experiment name).
        data_paths: List of image files or directories containing images
            for which predictions should be generated.
        root_dir: Directory containing saved training runs
            (default: ``runs``).

    Returns:
        list[str]: Predicted class labels for the provided inputs.

    Raises:
        ValueError: If the model directory, model file, or configuration
            file cannot be found.
    """
    root_path = Path(root_dir) / model_name
    if not root_path.exists():
        raise ValueError("This root directory path does not exist")

    model_path = root_path / f"{model_name}.pth"
    cfg_path = root_path / f"{model_name}.yaml"

    if not model_path.exists():
        raise ValueError(f"Model {model_name} not found.")
    if not cfg_path.exists():
        raise ValueError(f"Could not find a config file for a {model_name} model.")

    cfg = load_config([cfg_path])
    model = LeafCNN(cfg)
    model.load_state_dict(torch.load(model_path))
    pred_loader = load_inference_data(data_paths, cfg)

    preds = predict(model, pred_loader)

    return preds
    
    


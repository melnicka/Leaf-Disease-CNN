from pathlib import Path
import torch
import json
from .engine import train, score, predict, DEVICE
from .utils import training_setup, set_random_state, load_config
from .model import LeafCNN
from .dataloading import load_data, load_inference_data
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from .config_schema import Config

def train_model(name: str, cfg: Config):
    set_random_state(cfg)
    train_loader, val_loader, test_loader = load_data(cfg)
    writer = SummaryWriter(f"runs/tensorboards/{name}")
    model, optimizer, criterion, scheduler = training_setup(cfg)
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
    dir_path.mkdir(exist_ok=True)
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
    
    


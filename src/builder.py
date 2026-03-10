import os
import torch
import json
from src.engine import train, score, DEVICE
from src.utils import training_setup, set_random_state
from src.dataset import load_data
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
    dir_name = f"runs/{name}"
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), f"{dir_name}/{name}.pth")
    accuracy, precision, recall, f1 = score(model, test_loader)
    metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
    }
    with open(f"{dir_name}/eval_metrics.json", "w") as f:
        json.dump(metrics, f)

    OmegaConf.save(cfg, f"{dir_name}/{name}.yaml")
    

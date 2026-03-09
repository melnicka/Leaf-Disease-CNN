import torch
from src.engine import train, score, DEVICE
from src.utils import training_setup, set_random_state, load_config
from src.dataset import load_data
from torch.utils.tensorboard import SummaryWriter

# testing
if __name__ == '__main__':
    cfg = load_config()
    set_random_state(cfg)
    train_loader, val_loader, test_loader = load_data(cfg)
    writer = SummaryWriter(log_dir='src/runs')
    model, optimizer, criterion, scheduler  = training_setup(cfg)
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

    test_metrics = score(model, test_loader) 
    torch.save(model.state_dict(), "models/cnn0.pth")
    

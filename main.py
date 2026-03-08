from src.config.config import Config
from src.engine import train
from src.utils import training_setup
from src.dataset import load_data
from torch.utils.tensorboard import SummaryWriter

# testing
if __name__ == '__main__':
    cfg = Config()
    train_loader, val_loader, test_loader = load_data(cfg)
    writer = SummaryWriter(log_dir='src/runs')
    model, optimizer, criterion, scheduler  = training_setup(cfg)

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


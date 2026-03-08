from src.dataset import load_data
from src.config import DataConfig

# testing
if __name__ == '__main__':
    data_cfg = DataConfig()
    train_loader, val_loader, test_loader = load_data(data_cfg)
    print(train_loader)

from src.dataset import load_data
from src.config.config import DataConfig, ModelConfig
from src.model import LeafCNN
# testing
if __name__ == '__main__':
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_loader, val_loader, test_loader = load_data(data_cfg)
    
    model = LeafCNN(model_cfg)
    
    for x_batch, y_batch in val_loader:
        print(model(x_batch))


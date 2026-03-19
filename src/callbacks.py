import torch
import copy

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.no_improvement = 0
        self.best_loss = None
        self.best_model = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None or self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.no_improvement = 0
            model.load_state_dict(self.best_model)
            return False

        self.no_improvement +=1

        if self.no_improvement >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs.")
            return True

        return False


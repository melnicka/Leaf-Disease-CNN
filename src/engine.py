from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

if TYPE_CHECKING:
    from config_schema import Config
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> tuple [float, float]:
    """Trains one epoch.

    Args:
        model: The target classifier.
        train_loader: Training DataLoader.
        optimizer: Gradient-based optimizer.
        criterion: Loss function.

    Returns:
        tuple:
            - float: Training loss.
            - float: Training accuracy score.
    """
    running_loss = 0.0 
    running_correct = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(
                torch.argmax(y_pred, dim=1) == y_batch
        ).item()
        total_samples += y_batch.size(0)

    return running_loss / len(train_loader), running_correct / total_samples

def eval(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
) -> tuple[float, float, dict]:
    """Evaluates model on validation set during training.

    Args:
        model: The target classifier.
        val_loader: Validation DataLoader.
        criterion: Loss function.

    Returns:
        tuple:
        - float: Validation loss.
        - float: Validation accuracy score.
        - dict: Validation F1 score.
    """
    y_true, y_pred = [], []
    running_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            running_val_loss += criterion(preds, y_batch)

            preds = torch.argmax(preds, dim=1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(y_batch.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    val_loss = running_val_loss / len(val_loader)
    f1 = f1_score(y_true, y_pred, average=None)
    
    idx_to_class = val_loader.dataset.idx_to_class
    f1 = {idx_to_class[i]: score for i, score in enumerate(f1)}

    return val_loss, accuracy, f1

   
def train(
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: ReduceLROnPlateau,
        writer: SummaryWriter
):
    """Trains the model and logs the process into a tensorboard.

    Args:
        cfg: Configuration object.
        model: The target classifier.
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Gradient-based optimizer.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        writer: Tensorboard writer.
    """
    step = 0
    for epoch in range(cfg.train.num_epochs):
        print(f"\n====== Training epoch {epoch}... ======")
        print(f"Current learning rate: {scheduler.get_last_lr()}")
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)
        train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion
        )
        val_loss, val_acc, val_f1 = eval(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {100*train_acc:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {100*val_acc:.2f}%")
        val_f1_fmt = {k: f"{v:.4f}" for k, v in val_f1.items()}
        print(f"Validation F1 score: {val_f1_fmt}")

        writer.add_scalar('train/loss', train_loss, step)
        writer.add_scalar('train/accuracy', train_acc, step)
        writer.add_scalar('val/loss', val_loss, step)
        writer.add_scalar('val/accuracy', val_acc, step)
        writer.add_scalars('val/f1', val_f1), step
        step += 1

def score(
        model: nn.Module,
        test_loader: DataLoader,
) -> tuple[float, dict, dict, dict]:
    """Evaluates the model on the final test set.

    Args:
        model: The classifier to score.
        test_loader: Test DataLoader.

    Returns:
        tuple:
        - float: Accuracy score.
        - dict: Precision score for each class.
        - dict: Recall score for each class.
        - dict: F1 score for each class.
    """
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(DEVICE))

            preds = torch.argmax(preds, dim=1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(y_batch.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    
    idx_to_class = test_loader.dataset.idx_to_class
    f1 = {idx_to_class[i]: score for i, score in enumerate(f1)}
    precision = {idx_to_class[i]: score for i, score in enumerate(precision)}
    recall = {idx_to_class[i]: score for i, score in enumerate(recall)}

    f1_fmt = {k: f"{v:.4f}" for k, v in f1.items()}
    print(f"Validation F1 score: {f1_fmt}")
    precision_fmt = {k: f"{v:.4f}" for k, v in precision.items()}
    print(f"Precision score: {precision_fmt}")
    recall_fmt = {k: f"{v:.4f}" for k, v in recall.items()}
    print(f"Recall score: {recall_fmt}")

    return accuracy, precision, recall, f1

def predict(model: nn.Module, pred_loader: DataLoader) -> list:
    """Makes predictions.

    Args:
        model: A classifier for making predictions.
        pred_loader: DataLoader with predictions data (no labels).

    Returns:
        y_pred: The list of predicted classes.
    """
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x_batch in pred_loader:
            preds = model(x_batch.to(DEVICE))
            preds = torch.argmax(preds, dim=1)
            y_pred.extend(preds.cpu().tolist())

    

    return y_pred


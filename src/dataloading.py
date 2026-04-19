from __future__ import annotations
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from typing import TYPE_CHECKING
from .dataset import LeafImageDataset, LeafInferDataset

if TYPE_CHECKING:
    from numpy.typing.npt import ArrayLike
    from .config_schema import Config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_datasets(cfg: Config,) -> tuple[LeafImageDataset, ...]:
    """A pipeline for creating Dataset objects.

    Everything is controlled by the configuration object.

    Args:
        cfg: The configuration object.

    Returns:
        train-val-test split or train-test split. 
    """
    if not cfg.data.grayscale:
        train_transform = v2.Compose([
            v2.Resize(cfg.data.resize),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
            
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        eval_transform = v2.Compose([
            v2.Resize(cfg.data.resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
        ])
    else:
        train_transform = v2.Compose([
            v2.Resize(cfg.data.resize),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Grayscale(num_output_channels=1),
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5])
        ])

        eval_transform = v2.Compose([
            v2.Resize(cfg.data.resize),
            v2.Grayscale(num_output_channels=1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5])
        ])

    samples, labels, class_names, class_to_idx = collect_samples(cfg.data.root_dir)
    split = make_splits(
            samples,
            labels,
            cfg.data.test_size,
            cfg.data.val_size,
            cfg.random_state
    )

    if cfg.data.val_size is None:
        X_train, X_test, y_train, y_test = split 
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split
        val_dataset = LeafImageDataset(
                X_val,
                y_val,
                class_names,
                class_to_idx,
                eval_transform
        )
    train_dataset = LeafImageDataset(
            X_train,
            y_train,
            class_names,
            class_to_idx,
            train_transform
    )
    test_dataset = LeafImageDataset(
            X_test,
            y_test,
            class_names,
            class_to_idx,
            eval_transform
    )

    if cfg.data.val_size is None:
        return train_dataset, test_dataset

    return train_dataset, val_dataset, test_dataset


def load_inference_data(data_paths: list[str], cfg: Config) -> DataLoader:
    """A pipeline to create prediction Dataset object and loading it into a DataLoader object.

    Args:
        data_paths: Can contain paths to files, directories, or both.
        cfg: The same configuration object as the one used for the training. 

    Returns:
        DataLoader for making predictions.
    """
    if not cfg.data.grayscale:
        transform = v2.Compose([
            v2.Resize(cfg.data.resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    else:
        transform = v2.Compose([
            v2.Resize(cfg.data.resize),
            v2.Grayscale(num_output_channels=1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5])
        ])
    
    samples = collect_infer_samples(data_paths)
    dataset = LeafInferDataset(samples, transform)

    return DataLoader(dataset, cfg.data.batch_size, shuffle=False)

def collect_samples(root_dir: str = 'data') -> tuple[list[str], list[int], list[str], dict]:
    """Collects all data samples.

    Args:
        root_dir: Root data directory.

    Returns:
        tuple:
            - samples: Paths to all pictures.
            - labels: Numerical labels for all pictures. 
            - class_names: Name of each class.
            - class_to_idx: Class name to numerical label mapping.
    """
    root = Path(root_dir)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples =[]
    labels = []

    for class_name in class_names:
        for path in (root / class_name).iterdir():
            if path.is_file() and path.suffix.lower() in IMG_EXT:
                label = class_to_idx[class_name]
                samples.append(str(path))
                labels.append(label)

    return samples, labels, class_names, class_to_idx

def collect_infer_samples(data_paths: list[str]) -> list[str]:
    """Collects all samples for making predictions.

    Args:
        data_paths: Can contain paths to files, directories, or both.

    Returns:
        A list with paths to all pictures. 
    """
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []        

    for p in data_paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in IMG_EXT:
            samples.append(str(path))
        
        elif path.is_dir():
            for file in path.iterdir():
                if file.is_file() and file.suffix.lower() in IMG_EXT:
                    samples.append(str(file))

    return samples

def make_splits(
        samples: list[str],
        labels: list[int],
        test_size: float = 0.15,
        val_size: float | None = 0.1,
        random_state: int = 2137
) -> tuple[ArrayLike[str, int], ...]:

    """Creates train-val-test split or train-test split.

    Args:
        samples: Paths to all pictures.
        labels: Numerical labels for all pictures
        test_size: The size of the test set.
        val_size: The size og the validation set. If None, validation set will not be created.
        random_state: Random seed number.

    Returns:
        train-val-test split or train-test-split

    Raises:
        ValueError: If the split ratio do not add up to 1.
    """
    effective_val_size = 0.0 if val_size is None else val_size
    if 1 - test_size - effective_val_size <= 0:
        raise ValueError("Invalid train-test-split ratio.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
            samples,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
    )
    if val_size is None:
        return X_trainval, X_test, y_trainval, y_test
    
    val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size,
            stratify=y_trainval,
            random_state=random_state
            )

    return X_train, X_val, X_test, y_train, y_val, y_test


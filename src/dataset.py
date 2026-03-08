from pathlib import Path
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile
    from numpy.typing.npt import ArrayLike
    from .config import Config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class LeafImageDataset(Dataset):
    def __init__(
            self,
            samples: ArrayLike[str],
            labels: ArrayLike[int],
            class_names: ArrayLike[str],
            class_to_idx: dict[str, int],
            transform=None
    ):
        self.samples = samples
        self.labels = labels
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.idx_to_class = {i:c for c, i in class_to_idx.items()}
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> tuple[ImageFile, int]:
        image = Image.open(self.samples[index]).convert("RGB")
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

def load_data(cfg: Config) -> tuple[DataLoader, ...]:
    if cfg.data.grayscale is None:
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
            cfg.data.random_state
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
        val_loader = DataLoader(
                val_dataset,
                cfg.data.batch_size,
                shuffle=False
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
    train_loader = DataLoader(
            train_dataset,
            cfg.data.batch_size,
            shuffle=True,
    )
    test_loader = DataLoader(
            test_dataset,
            cfg.data.batch_size,
            shuffle=False
    )

    if cfg.data.val_size is None:
        return train_loader, test_loader

    return train_loader, val_loader, test_loader

def collect_samples(root_dir: str = 'data') -> tuple[list[str], list[int], list[str], dict]:
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

def make_splits(
        samples: list[str],
        labels: list[int],
        test_size: float = 0.15,
        val_size: float | None = 0.1,
        random_state: int = 2137
) -> tuple[ArrayLike[str], ...]:

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







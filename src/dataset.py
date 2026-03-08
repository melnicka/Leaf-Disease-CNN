from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile
    from numpy.typing.npt import ArrayLike

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class LeafImageDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> tuple[ImageFile, int]:
        image = Image.open(self.samples[index])
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

def collect_samples(root_dir: str = 'data') -> tuple[list[str], list[int], float, float]:
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
        samples: ArrayLike[str],
        labels: ArrayLike[int],
        test_size: float = 0.15,
        val_size: float = 0.1,
        random_state: int = 2137
        ):
    if 1 - test_size - val_size <= 0:
        raise ValueError("Ivalid train-test-split ratio.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
            samples,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
            )
    
    val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size,
            stratify=y_trainval,
            random_state=random_state
            )

    return X_train, X_val, X_test, y_train, y_val, y_test






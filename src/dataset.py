from __future__ import annotations
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile
    from numpy.typing.npt import ArrayLike

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
        self.weights = None
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[ImageFile, int]:
        image = Image.open(self.samples[index]).convert("RGB")
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def calc_class_weights(self):
        weights = []
        counter = Counter(self.labels)
        weight_sum = 0.0
        total_samples = sum(counter.values())
        print(sorted(counter))
        for label in sorted(counter):
            weight = total_samples / (7.0 * counter[label])
            weights.append(weight)
            weight_sum += weight

        weights = [float(np.round(w / weight_sum, 4)) for w in weights]

        self.weights = torch.tensor(weights)


class LeafInferDataset():
    def __init__(self, samples: ArrayLike, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> ImageFile:
        image = Image.open(self.samples[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image









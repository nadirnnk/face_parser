# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CelebAMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert mask to tensor (class indices, not one-hot encoding)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask

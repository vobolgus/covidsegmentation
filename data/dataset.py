"""COVID-19 CT Segmentation Dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        ToTensorV2(),
    ])


class CovidDataset(Dataset):
    """Dataset for COVID-19 CT segmentation."""

    def __init__(self, images: np.ndarray, masks: np.ndarray = None, transform=None):
        """
        Args:
            images: numpy array of shape (N, H, W) or (N, H, W, C)
            masks: numpy array of shape (N, H, W, 4) - 4 channels
            transform: albumentations transforms
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Normalize to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Ensure image is 2D for albumentations
        if len(image.shape) == 2:
            image = image
        elif len(image.shape) == 3:
            image = image[:, :, 0] if image.shape[2] > 1 else image.squeeze()

        if self.masks is not None:
            # Extract only channels 0 (ground glass) and 1 (consolidations)
            mask = self.masks[idx][:, :, :2].astype(np.float32)

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                image = torch.from_numpy(image)
                mask = torch.from_numpy(mask)

            # Ensure correct shapes: image (1, H, W), mask (2, H, W)
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
            if len(mask.shape) == 3:
                mask = mask.permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)

            return image.float(), mask.float()
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = torch.from_numpy(image)

            if len(image.shape) == 2:
                image = image.unsqueeze(0)

            return image.float()


def load_data():
    """Load all training data from numpy files."""
    images_medseg = np.load(f"{config.DATA_DIR}/images_medseg.npy")
    masks_medseg = np.load(f"{config.DATA_DIR}/masks_medseg.npy")
    images_radiopedia = np.load(f"{config.DATA_DIR}/images_radiopedia.npy")
    masks_radiopedia = np.load(f"{config.DATA_DIR}/masks_radiopedia.npy")

    # Combine datasets
    images = np.concatenate([images_medseg, images_radiopedia], axis=0)
    masks = np.concatenate([masks_medseg, masks_radiopedia], axis=0)

    return images, masks


def get_train_val_loaders(val_split=None, batch_size=None, seed=42):
    """Create train and validation data loaders."""
    val_split = val_split or config.VAL_SPLIT
    batch_size = batch_size or config.BATCH_SIZE

    images, masks = load_data()

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(images))
    split_idx = int(len(indices) * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = CovidDataset(
        images[train_indices],
        masks[train_indices],
        transform=get_train_transforms()
    )

    val_dataset = CovidDataset(
        images[val_indices],
        masks[val_indices],
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS works better with 0 workers
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


def get_test_loader(batch_size=None):
    """Create test data loader."""
    batch_size = batch_size or config.BATCH_SIZE

    test_images = np.load(f"{config.DATA_DIR}/test_images_medseg.npy")

    test_dataset = CovidDataset(
        test_images,
        masks=None,
        transform=get_val_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return test_loader

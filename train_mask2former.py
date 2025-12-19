"""Training script for COVID-19 CT segmentation using Mask2Former."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from tqdm import tqdm
import albumentations as A

import config

# Mask2Former config
MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
NUM_CLASSES = 2  # ground glass, consolidation
EPOCHS = 30
BATCH_SIZE = 4  # Mask2Former is memory intensive
LR = 5e-5


class CovidMask2FormerDataset(Dataset):
    """Dataset for Mask2Former."""

    def __init__(self, images, masks, processor, transform=None):
        self.images = images
        self.masks = masks
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Normalize image to [0, 255] uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert grayscale to RGB (Mask2Former expects RGB)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        # Create semantic mask: combine ground glass (0) and consolidation (1)
        # Background = 0, Ground glass = 1, Consolidation = 2
        semantic_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        semantic_mask[mask[:, :, 0] > 0.5] = 1  # Ground glass
        semantic_mask[mask[:, :, 1] > 0.5] = 2  # Consolidation (overwrites if overlap)

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=semantic_mask)
            image = transformed['image']
            semantic_mask = transformed['mask']

        # Process with Mask2Former processor
        inputs = self.processor(
            images=image,
            segmentation_maps=semantic_mask,
            return_tensors="pt",
            do_resize=True,
            size={"height": 512, "width": 512},
        )

        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        return inputs


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])


def collate_fn(batch):
    """Custom collate function for Mask2Former."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])

    # Handle mask labels and class labels
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


def load_data():
    """Load all training data."""
    images_medseg = np.load(f"{config.DATA_DIR}/images_medseg.npy")
    masks_medseg = np.load(f"{config.DATA_DIR}/masks_medseg.npy")
    images_radiopedia = np.load(f"{config.DATA_DIR}/images_radiopedia.npy")
    masks_radiopedia = np.load(f"{config.DATA_DIR}/masks_radiopedia.npy")

    images = np.concatenate([images_medseg, images_radiopedia], axis=0)
    masks = np.concatenate([masks_medseg, masks_radiopedia], axis=0)

    return images, masks


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0

    for batch in tqdm(loader, desc='Validation'):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        total_loss += outputs.loss.item()

    return total_loss / len(loader)


def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")

    # Load processor
    print("Loading processor...")
    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_NAME,
        do_resize=True,
        size={"height": 512, "width": 512},
        ignore_index=0,  # Background class
        reduce_labels=False,
    )

    # Load data
    print("Loading data...")
    images, masks = load_data()

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(images))
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create datasets
    train_dataset = CovidMask2FormerDataset(
        images[train_indices],
        masks[train_indices],
        processor,
        transform=get_train_transforms()
    )

    val_dataset = CovidMask2FormerDataset(
        images[val_indices],
        masks[val_indices],
        processor,
        transform=None
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model
    print("Loading model...")
    id2label = {0: "background", 1: "ground_glass", 2: "consolidation"}
    label2id = {v: k for k, v in id2label.items()}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=3,  # background + 2 classes
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')

    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("mask2former_best")
            processor.save_pretrained("mask2former_best")
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    print("-" * 60)
    print(f"Training complete! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

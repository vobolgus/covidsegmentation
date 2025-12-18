"""Training script for COVID-19 CT segmentation."""

import torch
from tqdm import tqdm

import config
from data import get_train_val_loaders
from models import create_model
from utils import CombinedLoss, calculate_metrics, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = []

    for images, masks in tqdm(loader, desc='Validation'):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        metrics = calculate_metrics(outputs, masks)
        all_metrics.append(metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    return avg_loss, avg_metrics


def main():
    print(f"Using device: {config.DEVICE}")
    print(f"Encoder: {config.ENCODER}")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_train_val_loaders()
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(config.DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_dice = 0
    patience_counter = 0

    print(f"\nStarting training for {config.EPOCHS} epochs (patience: {config.PATIENCE})...")
    print("-" * 60)

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, config.DEVICE)

        # Update scheduler
        scheduler.step()

        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"Ground Glass Dice: {val_metrics['ground_glass_dice']:.4f} | "
              f"Consolidation Dice: {val_metrics['consolidation_dice']:.4f} | "
              f"Mean Dice: {val_metrics['mean_dice']:.4f}")

        # Save best model
        if val_metrics['mean_dice'] > best_dice:
            best_dice = val_metrics['mean_dice']
            save_checkpoint(model, optimizer, epoch, val_loss, config.CHECKPOINT_PATH)
            print(f"Saved best model with Mean Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print("-" * 60)
    print(f"Training complete! Best Mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()

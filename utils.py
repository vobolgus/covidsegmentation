"""Utility functions for training and evaluation."""

import torch
import numpy as np
import matplotlib.pyplot as plt


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def f1_score(pred, target, threshold=0.5):
    """Calculate pixel-wise F1 score (same as Dice for binary)."""
    pred_binary = (pred > threshold).float()
    return dice_coefficient(pred_binary, target)


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate metrics for each class."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()

    metrics = {}
    class_names = ['ground_glass', 'consolidation']

    for i, name in enumerate(class_names):
        dice = dice_coefficient(pred_binary[:, i], target[:, i])
        metrics[f'{name}_dice'] = dice.item()

    # Average
    metrics['mean_dice'] = np.mean([metrics[f'{name}_dice'] for name in class_names])

    return metrics


class DiceLoss(torch.nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class TverskyLoss(torch.nn.Module):
    """Tversky loss - better for imbalanced segmentation."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = penalize missed detections more)
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


class FocalTverskyLoss(torch.nn.Module):
    """Focal Tversky loss - focuses on hard examples."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        return focal_tversky


class CombinedLoss(torch.nn.Module):
    """Combined Dice + BCE loss."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)


def load_checkpoint(model, path, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))


def visualize_predictions(images, masks, preds, num_samples=4, save_path=None):
    """Visualize model predictions."""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(min(num_samples, len(images))):
        # Image
        axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
        axes[i, 0].set_title('CT Image')
        axes[i, 0].axis('off')

        # Ground truth - ground glass
        axes[i, 1].imshow(masks[i, 0], cmap='Reds', alpha=0.7)
        axes[i, 1].set_title('GT: Ground Glass')
        axes[i, 1].axis('off')

        # Ground truth - consolidation
        axes[i, 2].imshow(masks[i, 1], cmap='Blues', alpha=0.7)
        axes[i, 2].set_title('GT: Consolidation')
        axes[i, 2].axis('off')

        # Prediction overlay
        pred_gg = (preds[i, 0] > 0.5).astype(float)
        pred_cons = (preds[i, 1] > 0.5).astype(float)
        overlay = np.zeros((*pred_gg.shape, 3))
        overlay[:, :, 0] = pred_gg  # Red for ground glass
        overlay[:, :, 2] = pred_cons  # Blue for consolidation
        axes[i, 3].imshow(images[i].squeeze(), cmap='gray')
        axes[i, 3].imshow(overlay, alpha=0.5)
        axes[i, 3].set_title('Prediction')
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

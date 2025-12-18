"""Inference and submission generation for COVID-19 CT segmentation."""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from data import get_test_loader, get_train_val_loaders
from models import create_model
from utils import load_checkpoint, f1_score


@torch.no_grad()
def predict(model, loader, device, use_tta=True):
    """Generate predictions for test data with optional TTA."""
    model.eval()
    all_preds = []

    for images in tqdm(loader, desc='Predicting'):
        images = images.to(device)

        if use_tta:
            # Test Time Augmentation: average predictions from multiple views
            preds = []

            # Original
            preds.append(torch.sigmoid(model(images)))

            # Horizontal flip
            flipped_h = torch.flip(images, dims=[3])
            pred_h = torch.sigmoid(model(flipped_h))
            preds.append(torch.flip(pred_h, dims=[3]))

            # Vertical flip
            flipped_v = torch.flip(images, dims=[2])
            pred_v = torch.sigmoid(model(flipped_v))
            preds.append(torch.flip(pred_v, dims=[2]))

            # Both flips
            flipped_hv = torch.flip(images, dims=[2, 3])
            pred_hv = torch.sigmoid(model(flipped_hv))
            preds.append(torch.flip(pred_hv, dims=[2, 3]))

            # Average all predictions
            pred = torch.stack(preds).mean(dim=0)
        else:
            pred = torch.sigmoid(model(images))

        all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


@torch.no_grad()
def find_best_threshold(model, val_loader, device, use_tta=True):
    """Find optimal threshold using validation set."""
    model.eval()
    all_preds = []
    all_masks = []

    print("Evaluating on validation set...")
    for images, masks in tqdm(val_loader, desc='Validation'):
        images = images.to(device)

        if use_tta:
            preds = []
            preds.append(torch.sigmoid(model(images)))

            flipped_h = torch.flip(images, dims=[3])
            pred_h = torch.sigmoid(model(flipped_h))
            preds.append(torch.flip(pred_h, dims=[3]))

            flipped_v = torch.flip(images, dims=[2])
            pred_v = torch.sigmoid(model(flipped_v))
            preds.append(torch.flip(pred_v, dims=[2]))

            flipped_hv = torch.flip(images, dims=[2, 3])
            pred_hv = torch.sigmoid(model(flipped_hv))
            preds.append(torch.flip(pred_hv, dims=[2, 3]))

            pred = torch.stack(preds).mean(dim=0)
        else:
            pred = torch.sigmoid(model(images))

        all_preds.append(pred.cpu())
        all_masks.append(masks)

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Try different thresholds
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = 0.5
    best_f1 = 0

    print("\nSearching for best threshold...")
    for thresh in thresholds:
        pred_binary = (all_preds > thresh).float()
        f1 = f1_score(pred_binary, all_masks).item()
        print(f"  Threshold {thresh:.2f}: F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"\nBest threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    return best_threshold


def create_submission(predictions, threshold=0.5, output_path=None):
    """Create submission file from predictions.

    Args:
        predictions: numpy array of shape (10, 2, 512, 512)
        threshold: threshold for binary prediction
        output_path: path to save submission CSV
    """
    output_path = output_path or config.SUBMISSION_PATH

    # Apply threshold
    pred_binary = (predictions > threshold).astype(int)

    # Reshape to (10, 512, 512, 2) as expected by competition
    pred_binary = pred_binary.transpose(0, 2, 3, 1)

    # Create submission DataFrame
    df = pd.DataFrame(
        data=np.stack(
            (np.arange(len(pred_binary.ravel())), pred_binary.ravel()),
            axis=-1
        ).astype(int),
        columns=['Id', 'Predicted']
    ).set_index('Id')

    df.to_csv(output_path)
    print(f"Submission saved to {output_path}")
    print(f"Shape: {pred_binary.shape}")
    print(f"Total pixels: {len(pred_binary.ravel())}")
    print(f"Positive predictions: {pred_binary.sum()} ({100 * pred_binary.mean():.2f}%)")

    return df


def main():
    print(f"Using device: {config.DEVICE}")

    # Load model
    print("Loading model...")
    model = create_model()
    model = model.to(config.DEVICE)

    # Load checkpoint
    epoch, val_loss = load_checkpoint(model, config.CHECKPOINT_PATH)
    print(f"Loaded checkpoint from epoch {epoch} with val_loss {val_loss:.4f}")

    # Find best threshold using validation set
    print("\nFinding optimal threshold...")
    _, val_loader = get_train_val_loaders()
    best_threshold = find_best_threshold(model, val_loader, config.DEVICE, use_tta=True)

    # Create test loader
    print("\nLoading test data...")
    test_loader = get_test_loader()
    print(f"Test samples: {len(test_loader.dataset)}")

    # Generate predictions with TTA
    print("Generating predictions with TTA...")
    predictions = predict(model, test_loader, config.DEVICE, use_tta=True)
    print(f"Predictions shape: {predictions.shape}")

    # Create submission with tuned threshold
    print(f"\nCreating submission with threshold {best_threshold:.2f}...")
    create_submission(predictions, threshold=best_threshold)

    print("\nDone! You can now submit to Kaggle:")
    print(f"  kaggle competitions submit -c covid-segmentation -f {config.SUBMISSION_PATH} -m 'U-Net with TTA'")


if __name__ == "__main__":
    main()

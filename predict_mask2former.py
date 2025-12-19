"""Inference script for Mask2Former COVID-19 segmentation."""

import numpy as np
import pandas as pd
import torch
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from tqdm import tqdm

import config

MODEL_PATH = "mask2former_best"


def load_test_data():
    """Load test images."""
    test_images = np.load(f"{config.DATA_DIR}/test_images_medseg.npy")
    return test_images


@torch.no_grad()
def predict(model, processor, images, device):
    """Generate predictions."""
    model.eval()
    all_preds = []

    for i in tqdm(range(len(images)), desc='Predicting'):
        image = images[i]

        # Normalize to uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert to RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        # Process image
        inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": 512, "width": 512},
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        outputs = model(**inputs)

        # Post-process to get semantic segmentation
        pred = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(512, 512)]
        )[0]

        # Convert to numpy
        pred = pred.cpu().numpy()

        # Convert semantic mask to 2-channel binary mask
        # Class 1 = ground glass, Class 2 = consolidation
        binary_mask = np.zeros((512, 512, 2), dtype=np.float32)
        binary_mask[:, :, 0] = (pred == 1).astype(np.float32)  # Ground glass
        binary_mask[:, :, 1] = (pred == 2).astype(np.float32)  # Consolidation

        all_preds.append(binary_mask)

    return np.array(all_preds)


def predict_with_tta(model, processor, images, device):
    """Predict with test-time augmentation."""
    model.eval()
    all_preds = []

    for i in tqdm(range(len(images)), desc='Predicting with TTA'):
        image = images[i]

        # Normalize to uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert to RGB
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=-1)

        preds = []

        # Original
        preds.append(predict_single(model, processor, image_rgb, device))

        # Horizontal flip
        flipped_h = np.fliplr(image_rgb).copy()
        pred_h = predict_single(model, processor, flipped_h, device)
        preds.append(np.fliplr(pred_h))

        # Vertical flip
        flipped_v = np.flipud(image_rgb).copy()
        pred_v = predict_single(model, processor, flipped_v, device)
        preds.append(np.flipud(pred_v))

        # Both flips
        flipped_hv = np.flipud(np.fliplr(image_rgb)).copy()
        pred_hv = predict_single(model, processor, flipped_hv, device)
        preds.append(np.fliplr(np.flipud(pred_hv)))

        # Average predictions
        avg_pred = np.mean(preds, axis=0)
        all_preds.append(avg_pred)

    return np.array(all_preds)


@torch.no_grad()
def predict_single(model, processor, image, device):
    """Predict for a single image."""
    inputs = processor(
        images=image,
        return_tensors="pt",
        do_resize=True,
        size={"height": 512, "width": 512},
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    pred = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(512, 512)]
    )[0]

    pred = pred.cpu().numpy()

    # Convert to 2-channel
    binary_mask = np.zeros((512, 512, 2), dtype=np.float32)
    binary_mask[:, :, 0] = (pred == 1).astype(np.float32)
    binary_mask[:, :, 1] = (pred == 2).astype(np.float32)

    return binary_mask


def create_submission(predictions, threshold=0.5, output_path="submission_mask2former.csv"):
    """Create submission file."""
    pred_binary = (predictions > threshold).astype(int)

    # Shape should be (10, 512, 512, 2)
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
    device = config.DEVICE
    print(f"Using device: {device}")

    # Load model and processor
    print("Loading model...")
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_PATH)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_PATH)
    model = model.to(device)

    # Load test data
    print("Loading test data...")
    test_images = load_test_data()
    print(f"Test samples: {len(test_images)}")

    # Predict with TTA
    print("Generating predictions with TTA...")
    predictions = predict_with_tta(model, processor, test_images, device)
    print(f"Predictions shape: {predictions.shape}")

    # Create submission
    print("Creating submission...")
    create_submission(predictions, threshold=0.5)

    print("\nDone! Submit with:")
    print("  kaggle competitions submit -c covid-segmentation -f submission_mask2former.csv -m 'Mask2Former'")


if __name__ == "__main__":
    main()

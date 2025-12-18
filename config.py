"""Configuration and hyperparameters for COVID-19 CT segmentation."""

import torch

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Data
DATA_DIR = "."
IMG_SIZE = 512
NUM_CLASSES = 2  # ground glass, consolidations

# Training
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2

# Model
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# Early stopping
PATIENCE = 15

# Paths
CHECKPOINT_PATH = "best_model.pth"
SUBMISSION_PATH = "submission.csv"

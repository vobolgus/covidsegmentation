"""U-Net model for COVID-19 CT segmentation."""

import segmentation_models_pytorch as smp
import config


def create_model():
    """Create U-Net model with pretrained encoder."""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=1,  # Grayscale CT images
        classes=config.NUM_CLASSES,  # ground glass + consolidations
    )
    return model

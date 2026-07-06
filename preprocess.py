from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


# Exact validation/test transform from the Kaggle notebook.
inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),
    ]
)


def validate_bgr_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("The input image must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("The MRI image must have three color channels.")

    if image_bgr.size == 0:
        raise ValueError("The MRI image is empty.")


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR image to the notebook inference tensor."""
    validate_bgr_image(image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    tensor = inference_transform(pil_image)
    return tensor.unsqueeze(0)

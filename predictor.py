from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from config import MODEL_CLASS_NAMES, DISPLAY_CLASS_NAMES, MODEL_NAME
from preprocess import preprocess_image


def predict(model, device, image_bgr: np.ndarray) -> dict[str, Any]:

    input_tensor = preprocess_image(image_bgr).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (time.perf_counter() - start) * 1000

    conf, idx = torch.max(probs, 1)

    idx = int(idx.item())
    conf = float(conf.item() * 100)

    probabilities = {
        DISPLAY_CLASS_NAMES[i]: float(probs[0][i].item() * 100)
        for i in range(len(DISPLAY_CLASS_NAMES))
    }

    return {
        "predicted_index": idx,
        "model_class": MODEL_CLASS_NAMES[idx],
        "display_class": DISPLAY_CLASS_NAMES[idx],
        "confidence": conf,
        "probabilities": probabilities,
        "inference_time_ms": inference_time,
        "model_name": MODEL_NAME,
        "input_tensor": input_tensor,
    }
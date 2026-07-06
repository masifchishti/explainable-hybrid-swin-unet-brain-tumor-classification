from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from config import HISTORY_FILE, HISTORY_IMAGES_DIR


def _read_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.is_file():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def load_history() -> list[dict[str, Any]]:
    return list(reversed(_read_history()))


def save_prediction(
    image_rgb: np.ndarray,
    original_filename: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    HISTORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    record_id = uuid4().hex
    image_path = HISTORY_IMAGES_DIR / f"{record_id}.png"
    Image.fromarray(image_rgb).save(image_path)

    now = datetime.now()

    record = {
        "id": record_id,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "source_file": original_filename,
        "image_path": str(image_path),
        "prediction": result["display_class"],
        "model_class": result["model_class"],
        "confidence": round(float(result["confidence"]), 4),
        "inference_time_ms": round(
            float(result["inference_time_ms"]),
            4,
        ),
        "probabilities": {
            key: round(float(value), 4)
            for key, value in result["probabilities"].items()
        },
    }

    history = _read_history()
    history.append(record)

    temporary_path = HISTORY_FILE.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    os.replace(temporary_path, HISTORY_FILE)
    return record


def clear_history() -> None:
    if HISTORY_FILE.is_file():
        HISTORY_FILE.unlink()

    if HISTORY_IMAGES_DIR.is_dir():
        for file_path in HISTORY_IMAGES_DIR.glob("*.png"):
            file_path.unlink(missing_ok=True)

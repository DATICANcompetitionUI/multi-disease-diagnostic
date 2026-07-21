import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from backend.services.downloader import download_file
from backend.services.image_processor import preprocess_image

# ============================================
# Google Drive File ID
# ============================================

MODEL_FILE_ID = "1rAHXl5a687eqoschSwWCNpXi48WNk2-Z"

# ============================================
# Paths
# ============================================

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "malaria_model.keras",
    )
)

LABELS_PATH = Path(
    os.getenv(
        "LABELS_PATH",
        "class_labels.json",
    )
)

# Download model if it doesn't exist
download_file(
    MODEL_FILE_ID,
    str(MODEL_PATH),
)


def resolve_model_path():

    env_model = os.getenv("MODEL_PATH")

    if env_model:

        candidate = Path(env_model)

        if candidate.exists():
            return candidate

    for candidate in (
        MODEL_PATH,
        Path("malaria_model.h5"),
        Path("malaria_model.keras"),
    ):
        if candidate.exists():
            return candidate

    return MODEL_PATH


def load_class_labels() -> List[str]:

    if LABELS_PATH.exists():

        with LABELS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if isinstance(data, dict) and "class_names" in data:
            return data["class_names"]

        if isinstance(data, list):
            return data

    return []


@lru_cache(maxsize=1)
def load_model():

    model_path = resolve_model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            "No malaria model file found."
        )

    return tf.keras.models.load_model(
        str(model_path)
    )


def predict_class_from_probabilities(
    probabilities,
    class_names,
):

    probs = np.asarray(probabilities).reshape(-1)

    if probs.size == 0:
        raise ValueError(
            "Model returned no probabilities."
        )

    if len(class_names) == 2 and probs.size == 1:

        probability = float(probs[0])

        predicted_index = (
            1 if probability >= 0.5 else 0
        )

        return (
            class_names[predicted_index],
            probability,
        )

    predicted_index = int(np.argmax(probs))

    return (
        class_names[predicted_index],
        float(probs[predicted_index]),
    )


def get_health_status():

    return {
        "status": "ok",
        "model": str(resolve_model_path()),
        "labels": str(LABELS_PATH),
    }


def predict_malaria(image_bytes: bytes):

    image = preprocess_image(image_bytes)

    model = load_model()

    probabilities = model.predict(
        image,
        verbose=0,
    )

    class_names = load_class_labels()

    predicted_class, probability = (
        predict_class_from_probabilities(
            probabilities,
            class_names,
        )
    )

    confidence = (
        max(probability, 1 - probability)
        if len(class_names) == 2
        else probability
    )

    return {
        "predicted_class": predicted_class,
        "probability": round(float(probability), 4),
        "confidence": round(float(confidence), 4),
        "model_path": str(resolve_model_path()),
    }
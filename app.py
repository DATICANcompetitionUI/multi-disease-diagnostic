import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from pydantic import BaseModel

import os
import gdown

def download_model():
    if not os.path.exists("malaria_model.keras"):
        print("Downloading model...")
        gdown.download(
            "https://drive.google.com/file/d/1rAHXl5a687eqoschSwWCNpXi48WNk2-Z/view?usp=drivesdk",
            "malaria_model.keras",
            quiet=False
        )

download_model()


MODEL_PATH = Path(os.getenv("MODEL_PATH", "malaria_model.keras"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "class_labels.json"))
IMG_SIZE = tuple(
    int(value) for value in os.getenv("IMAGE_SIZE", "64,64").split(",") if value.strip()
)
if len(IMG_SIZE) != 2:
    IMG_SIZE = (64, 64)

app = FastAPI(title="Image Classification API", version="1.0.0")


class PredictionResponse(BaseModel):
    predicted_class: str
    probability: float
    confidence: float
    model_path: str


def resolve_model_path() -> Path:
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        candidate = Path(env_model)
        if candidate.exists():
            return candidate

    for candidate in [MODEL_PATH, Path("malaria_model.h5"), Path("malaria_model.keras")]:
        if candidate.exists():
            return candidate
    return MODEL_PATH


def load_class_labels() -> List[str]:
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "class_names" in data:
            return list(data["class_names"])
        if isinstance(data, list):
            return list(data)
    return []


@lru_cache(maxsize=1)
def load_model():
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            "No model file found. Expected one of: "
            f"{MODEL_PATH}, malaria_model.h5, or malaria_model.keras. "
            "Run 'python train_model.py' first or set MODEL_PATH."
        )
    return tf.keras.models.load_model(str(model_path))


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = ImageOps.fit(image, IMG_SIZE, method=Image.Resampling.LANCZOS)
    image = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def predict_class_from_probabilities(probabilities: np.ndarray, class_names: List[str]) -> tuple[str, float]:
    probs = np.asarray(probabilities).reshape(-1)
    if probs.size == 0:
        raise ValueError("Model returned no probabilities.")

    if len(class_names) == 2 and probs.size == 1:
        probability = float(probs[0])
        predicted_index = 1 if probability >= 0.5 else 0
        return class_names[predicted_index] if predicted_index < len(class_names) else str(predicted_index), probability

    if len(class_names) > 1 and probs.size == len(class_names):
        predicted_index = int(np.argmax(probs))
        return class_names[predicted_index], float(probs[predicted_index])

    predicted_index = int(np.argmax(probs))
    class_name = class_names[predicted_index] if predicted_index < len(class_names) else f"class_{predicted_index}"
    return class_name, float(probs[predicted_index])


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "model": str(resolve_model_path()),
        "labels": str(LABELS_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        model = load_model()
        raw_output = model.predict(image_array, verbose=0)
        probabilities = np.asarray(raw_output).reshape(-1)
        class_names = load_class_labels()
        predicted_class, probability = predict_class_from_probabilities(probabilities, class_names)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    confidence = float(max(probability, 1.0 - probability)) if len(class_names) == 2 else float(probability)

    return PredictionResponse(
        predicted_class=predicted_class,
        probability=round(float(probability), 4),
        confidence=round(confidence, 4),
        model_path=str(resolve_model_path()),
    )

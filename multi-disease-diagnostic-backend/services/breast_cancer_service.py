import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import tensorflow as tf

from services.downloader import download_file

# ==================================================
# Prediction Labels
# ==================================================

LABELS = ["Benign", "Malignant"]

# ==================================================
# Google Drive File IDs
# ==================================================

MODEL_FILE_ID = "1xdWk3cNLTFPbyV56LhUxk_XCGdqQ2ok2"
SCALER_FILE_ID = "1h0K9LX6jy86m8VV-KrwVKfh03RkraJQP"
FEATURE_NAMES_FILE_ID = "1OUFXn8zhObGja2TKVZOhC1nbm2gMThkQ"


# ==================================================
# File Paths
# ==================================================

MODEL_PATH = Path(
    os.getenv(
        "BREAST_CANCER_MODEL_PATH",
        "breast_cancer_model.keras",
    )
)

SCALER_PATH = Path(
    os.getenv(
        "BREAST_CANCER_SCALER_PATH",
        "breast_cancer_scaler.pkl",
    )
)

FEATURES_PATH = Path(
    os.getenv(
        "FEATURE_NAMES_PATH",
        "feature_names.json",
    )
)


# ==================================================
# Download Required Files
# ==================================================

download_file(MODEL_FILE_ID, str(MODEL_PATH))
download_file(SCALER_FILE_ID, str(SCALER_PATH))
download_file(FEATURE_NAMES_FILE_ID, str(FEATURES_PATH))


# ==================================================
# Cached Loaders
# ==================================================

@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def load_scaler():
    return joblib.load(SCALER_PATH)


@lru_cache(maxsize=1)
def load_feature_names():

    with FEATURES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)




# ==================================================
# Health Endpoint
# ==================================================

def get_health_status():

    return {
    "status": "ok",
    "model": str(MODEL_PATH),
    "scaler": str(SCALER_PATH),
    "feature_names": str(FEATURES_PATH),
}


# ==================================================
# Prediction
# ==================================================

def predict_breast_cancer(
    features: Dict[str, float],
):

    feature_names = load_feature_names()

    missing = [
        feature
        for feature in feature_names
        if feature not in features
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    values = np.array(
        [
            [
                float(features[name])
                for name in feature_names
            ]
        ],
        dtype=np.float32,
    )

    scaler = load_scaler()

    values = scaler.transform(values)

    model = load_model()

    probability = float(
        model.predict(
            values,
            verbose=0,
        )[0][0]
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )


    confidence = (
        probability
        if prediction == 1
        else 1 - probability
    )

    return {
        "prediction": LABELS[prediction],
        "probability": round(
            probability,
            4,
        ),
        "confidence": round(
            confidence,
            4,
        ),
        "model_path": str(MODEL_PATH),
    }
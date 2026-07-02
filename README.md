# Malaria Detection API

This project provides a lightweight image classification service for detecting malaria from microscopy images. It uses a pre-trained Keras model and exposes a FastAPI endpoint that accepts an uploaded image and returns the predicted class along with a confidence score.

## What this project does

- Loads a trained malaria classification model from the repository
- Preprocesses uploaded images to the model input size
- Returns a prediction such as "Parasitized" or "Uninfected"
- Includes a training script for retraining the model from a dataset

## Project structure

- app.py: FastAPI application and prediction endpoint
- train_model.py: training script for building a new model
- malaria_model.keras: pre-trained model weights
- class_labels.json: label mapping for the model
- data/: dataset archive and related assets

## Requirements

- Python 3.10+
- pip
- A CPU or GPU capable of running TensorFlow

For most local setups, the CPU-only TensorFlow package is sufficient.

## Setup on Windows

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Verify that the model file exists:

   ```powershell
   dir malaria_model.keras
   ```

5. Start the API server:

   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

6. Open the interactive API docs in your browser:

   - Swagger UI: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc

## Test the API

You can upload an image using the Swagger UI or by running a command such as:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample-image.png"
```

Example response:

```json
{
  "predicted_class": "Parasitized",
  "probability": 0.9432,
  "confidence": 0.9432,
  "model_path": "malaria_model.keras"
}
```

## Training a new model

If you want to retrain the classifier from a dataset:

1. Prepare your dataset in the expected folder structure:

   ```text
   data/
     cell_images/
       Parasitized/
       Uninfected/
   ```

2. If the dataset is packaged as a zip file, extract it first.
3. Run:

   ```powershell
   python train_model.py
   ```

4. The script will save a new model to `malaria_model.keras`.

## Environment variables

You can override default paths and image size with environment variables:

- MODEL_PATH: path to the model file
- LABELS_PATH: path to the labels JSON file
- IMAGE_SIZE: image size as `width,height` (default: `64,64`)

Example:

```powershell
$env:MODEL_PATH = "malaria_model.keras"
$env:LABELS_PATH = "class_labels.json"
$env:IMAGE_SIZE = "64,64"
```

## Troubleshooting

- If the server fails to start, confirm that TensorFlow is installed correctly.
- If the model is missing, run `python train_model.py` or set `MODEL_PATH` to the correct file.
- If image predictions fail, make sure the uploaded file is a valid image and not empty.
- If you are on a machine with limited memory, reduce the batch size in `train_model.py`.

## Notes

The included model is intended for demonstration and local experimentation. For production use, you should validate it on a representative dataset and review its performance before deployment.

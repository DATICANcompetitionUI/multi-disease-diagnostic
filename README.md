# MedAI Nigeria 🏥

> **AI-Powered Multi-Disease Diagnostic Platform for Nigerian Healthcare**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://tensorflow.org)

---

## 🏆 Competition

**NACOS University of Ibadan × DATICAN**
Undergraduate Students' Competition in the Application of Artificial Intelligence in Medicine — 2026

**Team:** InsightLab
**Institution:** University of Ibadan, Nigeria
**Faculty:** Computing

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| **Backend API** | `https://multi-disease-diagnostic.onrender.com` |
| **Frontend App** | *https://multi-disease-diagnostic-frontends.onrender.com* |
| **API Documentation** | `https://multi-disease-diagnostic.onrender.com/docs` |

---

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Modules](#disease-modules)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Overview

MedAI Nigeria is a web-based AI diagnostic platform that enables Nigerian clinicians, community health workers, and medical students to rapidly screen for multiple diseases using artificial intelligence. The platform is designed to be accessible on any device — from smartphones in rural health centres to desktop computers in teaching hospitals.

### Why MedAI Nigeria?

Nigeria faces a critical healthcare challenge:
- **Doctor-to-patient ratio:** 4 doctors per 10,000 people (WHO recommends 10)
- **Annual malaria cases:** Over 60 million reported annually — Nigeria accounts for ~27% of global malaria burden
- **Breast cancer:** Leading female cancer in Nigeria with late-stage diagnosis as a major challenge due to limited screening infrastructure
- **Rural access:** Over 50% of Nigerians live in rural areas with limited access to specialist care

MedAI Nigeria addresses this gap by bringing AI-powered screening directly to a browser — no installation required, no specialist needed at point of care.

---

## Disease Modules

### ✅ Module 1: Malaria Detection (Live)

Detects malaria parasites from blood smear microscopy images using deep learning.

- **Model:** MobileNetV2 (Transfer Learning)
- **Dataset:** NIH/Kaggle Malaria Cell Images Dataset — 27,558 labeled cell images
- **Classes:** Parasitized | Uninfected
- **Input:** Blood smear microscopy image (PNG, JPG, TIFF)
- **Output:** Classification result + confidence score

**Clinical relevance:** Manual microscopy analysis is time-consuming, requires trained technicians, and is prone to fatigue-related errors. This module provides consistent, instant second-opinion screening.

---

### ✅ Module 2: Breast Cancer Detection (Live)

Classifies breast tumour cell nuclei measurements as benign or malignant using a neural network.

- **Model:** Keras Neural Network
- **Dataset:** Wisconsin Breast Cancer Dataset (UCI Machine Learning Repository)
- **Classes:** Benign | Malignant
- **Input:** 30 numerical cell nucleus measurements from Fine Needle Aspiration (FNA)
- **Output:** Classification result + probability + confidence score

**Clinical relevance:** Fine Needle Aspiration is a standard breast cancer diagnostic procedure. AI-assisted classification reduces pathologist workload and supports faster diagnosis in resource-limited settings.

---

### 🔜 Coming Soon

| Module | Description | Status |
|---|---|---|
| Diabetes Risk | Risk prediction from clinical parameters | In Development |
| Skin Disease Detection | Dermatological condition classification from images | In Development |
| Chest X-Ray Analysis | Pneumonia and TB detection from chest X-rays | Planned |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend                          │
│              React · Space Grotesk · Teal UI         │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP/REST
┌─────────────────────▼───────────────────────────────┐
│                  FastAPI Backend                     │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Routes  │  │ Schemas  │  │    Services      │   │
│  │ malaria  │  │ malaria  │  │  malaria_service │   │
│  │ breast   │  │ breast   │  │  breast_service  │   │
│  │ cancer   │  │ cancer   │  │  downloader      │   │
│  └──────────┘  └──────────┘  │  image_processor │   │
│                               └──────────────────┘   │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼───────┐         ┌─────────▼──────────┐
│ TensorFlow    │         │  Keras Neural Net  │
│ MobileNetV2   │         │  + StandardScaler  │
│ Malaria Model │         │  Breast Cancer     │
│ (.keras)      │         │  Model (.keras)    │
└───────────────┘         └────────────────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
              ┌───────▼───────┐
              │  Google Drive │
              │  Model Store  │
              │  (Auto-DL)    │
              └───────────────┘
```

---

## Project Structure

```
multi-disease-diagnostic/
│
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for Render
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── multi-disease-diagnostic-backend/
├ ├── app.py                          # FastAPI application entry point
│
├── routes/                         # API route handlers
│   ├── __init__.py
│   ├── malaria.py                  # POST /malaria/predict
│   └── breast_cancer.py            # POST /breast-cancer/predict
│
├── schemas/                        # Pydantic request/response models
│   ├── __init__.py
│   ├── malaria.py                  # MalariaResponse schema
│   └── breast_cancer.py            # BreastCancerRequest/Response schemas
│
├── services/                       # Business logic layer
│   ├── __init__.py
│   ├── malaria_service.py          # Malaria prediction logic
│   ├── malaria_model.py            # Malaria model loader
│   ├── breast_cancer_service.py    # Breast cancer prediction logic
│   ├── image_processor.py          # Image preprocessing utilities
│   └── downloader.py               # Google Drive model downloader
│
├── models/                         # Model files (auto-downloaded)
│   ├── malaria_model.keras         # MobileNetV2 malaria classifier
│   ├── breast_cancer_model.keras   # Neural network classifier
│   ├── breast_cancer_scaler.pkl    # StandardScaler for breast cancer
│   └── feature_names.json          # Feature name mapping
│

```

---

## Tech Stack

### Backend
| Technology | Purpose | Version |
|---|---|---|
| Python | Core language | 3.11+ |
| FastAPI | REST API framework | 0.110+ |
| Uvicorn | ASGI server | 0.27+ |
| TensorFlow | Deep learning (malaria + breast cancer) | 2.20 |
| Keras | Neural network API | Built into TF |
| Scikit-learn | ML utilities | 1.4+ |
| Pillow | Image processing | 10.0+ |
| NumPy | Numerical computing | 1.24+ |
| Pandas | Data manipulation | 2.0+ |
| Joblib | Model serialization | 1.3+ |
| gdown | Google Drive downloads | 5.2+ |

### Frontend
| Technology | Purpose |
|---|---|
| React | UI framework |
| Space Grotesk + Inter | Typography |


### Infrastructure
| Service | Purpose |
|---|---|
| Render | Backend hosting (auto-deploy from GitHub) |
| Google Drive | Model file storage |
| GitHub | Version control + CI/CD trigger |

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/DATICANcompetitionUI/multi-disease-diagnostic.git
cd multi-disease-diagnostic
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the development server

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

> **Note:** On first startup, model files are automatically downloaded from Google Drive. This may take 1–2 minutes depending on your connection speed.

### 5. Open API documentation

Navigate to `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## API Documentation

### Base URL
```
https://multi-disease-diagnostic.onrender.com
```

---

### `GET /health`

Returns the health status of the API and confirms all model files are loaded.

**Response:**
```json
{
  "status": "ok",
  "malaria_model": "models/malaria_model.keras",
  "breast_cancer_model": "models/breast_cancer_model.keras"
}
```

---

### `POST /malaria/predict`

Classifies a blood smear image as parasitized (malaria positive) or uninfected.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | Image file | Blood smear microscopy image (PNG, JPG, TIFF) |

**Example (curl):**
```bash
curl -X POST "https://multi-disease-diagnostic.onrender.com/malaria/predict" \
  -F "file=@blood_smear.png"
```

**Response:**
```json
{
  "predicted_class": "Parasitized",
  "probability": 0.9731,
  "confidence": 0.9731,
  "model_path": "models/malaria_model.keras"
}
```

| Field | Type | Description |
|---|---|---|
| `predicted_class` | string | `"Parasitized"` or `"Uninfected"` |
| `probability` | float | Raw model output (0–1) |
| `confidence` | float | Confidence in the prediction (0–1) |
| `model_path` | string | Model file used for prediction |

---

### `POST /breast-cancer/predict`

Classifies breast tumour cell measurements as benign or malignant.

**Request:** `application/json`

```json
{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```

**Response:**
```json
{
  "prediction": "Malignant",
  "probability": 0.9823,
  "confidence": 0.9823
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | string | `"Benign"` or `"Malignant"` |
| `probability` | float | Raw model output (0–1) |
| `confidence` | float | Confidence in the prediction (0–1) |

---

### Error Responses

| Code | Meaning | Common Cause |
|---|---|---|
| `400` | Bad Request | Missing or invalid features |
| `422` | Unprocessable Content | Wrong data types in request body |
| `500` | Internal Server Error | Model not loaded or server issue |

---

## Model Details

### Malaria Detection Model

| Property | Value |
|---|---|
| Architecture | MobileNetV2 + Custom Head |
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Input Size | 64 × 64 × 3 |
| Output | Sigmoid (binary) |
| Training Dataset | NIH Malaria Cell Images (Kaggle) |
| Dataset Size | 27,558 images |
| Classes | Parasitized, Uninfected |
| Training Epochs | 8 |
| Optimizer | Adam (lr=0.0001) |
| Loss | Binary Crossentropy |
| Training Accuracy | *see training logs* |
| Validation Accuracy | *see training logs* |

**Model Head:**
```
MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
```

---

### Breast Cancer Detection Model

| Property | Value |
|---|---|
| Architecture | Keras Neural Network |
| Input Features | 30 numerical cell nucleus measurements |
| Output | Sigmoid (binary) |
| Training Dataset | Wisconsin Breast Cancer Dataset (UCI) |
| Classes | Benign, Malignant |
| Preprocessing | StandardScaler normalization |

**Features used (30 total):**
Cell nucleus measurements computed from digitized Fine Needle Aspiration (FNA) images — mean, standard error, and worst values for: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

---

## Deployment

### Render (Production)

The backend is deployed on Render with automatic deployments triggered by GitHub pushes.

**Configuration:**

| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Python Version | 3.11 (via `runtime.txt`) |
| Instance Type | Free |

**Model Loading Strategy:**
Models are stored on Google Drive and automatically downloaded on first startup using `gdown`. This avoids GitHub's 100MB file size limit.

```python
# Automatic model download on startup
download_file(MODEL_FILE_ID, str(MODEL_PATH))
```

### Frontend 

The React frontend is deployed on Render and connects to the Render backend API.

---

## Training Your Own Models

### Malaria Detection

```bash
# Download dataset from Kaggle first
python train_model.py \
  --data-dir data \
  --output-model models/malaria_model.keras \
  --epochs 8 \
  --image-size 64 64
```

**Available flags:**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Path to dataset root |
| `--output-model` | `malaria_model.keras` | Output model path |
| `--metadata-file` | `class_labels.json` | Class labels output |
| `--image-size` | `64 64` | Input image dimensions |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--validation-split` | `0.2` | Validation split ratio |

---

## CORS Configuration

The API is configured to accept requests from all origins to support frontend integration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Ethical Considerations

- All predictions include a **clinical disclaimer** — results are for screening purposes only and must not replace professional medical diagnosis
- Models are trained on publicly available, ethically sourced datasets
- No patient data is stored by the platform
- The platform is designed to **assist** clinicians, not replace them

---

## Roadmap

- [x] Malaria detection from blood smear images
- [x] Breast cancer classification from cell measurements
- [x] Modular codebase architecture
- [x] Auto model download from Google Drive
- [x] Deployed backend on Render
- [x] Responsive frontend on Lovable
- [ ] Diabetes risk prediction module
- [ ] Skin disease detection module
- [ ] Chest X-ray analysis module
- [ ] Patient session history
- [ ] PDF diagnostic report export
- [ ] Offline-capable Progressive Web App (PWA)

---

## Team

**Team InsightLab — University of Ibadan**

| Member | Role |
|---|---|
| Ayoola Nifemi | Frontend Developer |
| Covenant Ekundayo | Mobile & Backend Development, UI/UX |

**GitHub:** [github.com/DATICANcompetitionUI/multi-disease-diagnostic](https://github.com/DATICANcompetitionUI/multi-disease-diagnostic)

**Behance (UI/UX Portfolio):** [behance.net/covenantekundayo](https://behance.net/covenantekundayo)

---

## License

This project was built for the NACOS UI × DATICAN AI in Medicine Competition 2026 and is intended for educational and research purposes.

---

## Acknowledgements

- **NACOS University of Ibadan** and **DATICAN** for organizing this competition
- **NIH / Kaggle** for the Malaria Cell Images Dataset
- **UCI Machine Learning Repository** for the Wisconsin Breast Cancer Dataset
- **Google** for MobileNetV2 architecture and TensorFlow
- **FastAPI** team for the excellent Python web framework

---

> ⚠️ **Medical Disclaimer:** MedAI Nigeria is an AI-assisted screening tool developed for research and educational purposes. All diagnostic outputs should be verified by a qualified healthcare professional. Do not make clinical decisions based solely on this platform's results.

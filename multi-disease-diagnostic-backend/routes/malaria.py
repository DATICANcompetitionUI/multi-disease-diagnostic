from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from schemas.malaria import PredictionResponse
from services.malaria_service import (
    get_health_status,
    predict_malaria,
)

router = APIRouter()


@router.get("/health")
def health():

    return get_health_status()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    try:

        image_bytes = await file.read()

        return predict_malaria(
            image_bytes
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image upload: {exc}",
        )
from fastapi import APIRouter, HTTPException

from backend.schemas.breast_cancer import (
    BreastCancerRequest,
    BreastCancerResponse,
)

from backend.services.breast_cancer_service import (
    predict_breast_cancer,
    get_health_status,
)

router = APIRouter()


@router.get("/health")
def health():
    return get_health_status()


@router.post(
    "/predict",
    response_model=BreastCancerResponse,
)
def predict(data: BreastCancerRequest):

    try:
        return predict_breast_cancer(data.features)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
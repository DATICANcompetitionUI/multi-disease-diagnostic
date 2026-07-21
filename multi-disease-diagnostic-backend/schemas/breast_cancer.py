from pydantic import BaseModel
from typing import Dict, Union


class BreastCancerRequest(BaseModel):
    features: Dict[str, Union[str, float, int]]


class BreastCancerResponse(BaseModel):
    prediction: str
    probability: float
    confidence: float
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    predicted_class: str
    probability: float
    confidence: float
    model_path: str
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, examples=["Summarize this article for me."])


class PredictResponse(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class CategoriesResponse(BaseModel):
    categories: list[str]

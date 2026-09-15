"""FastAPI inference service for the AI Request Dispatcher."""
import json
import logging
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings
from app.schemas import CategoriesResponse, HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

state = {"model": None, "tokenizer": None, "id2cat": None, "device": None}


def _resolve_model_source() -> str:
    """Returns the local path or HF Hub repo id to load the model/tokenizer from."""
    if settings.model_source == "hf_hub":
        if not settings.hf_model_repo:
            raise RuntimeError("MODEL_SOURCE=hf_hub requires HF_MODEL_REPO to be set")
        return settings.hf_model_repo
    return settings.model_dir


def load_model():
    source = _resolve_model_source()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from '%s' onto %s", source, device)

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to(device)
    model.eval()

    id2cat_path = os.path.join(settings.model_dir, "id2cat.json")
    if settings.model_source == "local" and os.path.exists(id2cat_path):
        with open(id2cat_path) as f:
            id2cat = {int(k): v for k, v in json.load(f).items()}
    else:
        # Fall back to the model config's label mapping (works for hf_hub too,
        # as long as id2label was saved with save_pretrained()).
        id2cat = {int(k): v for k, v in model.config.id2label.items()}

    state.update(model=model, tokenizer=tokenizer, id2cat=id2cat, device=device)
    logger.info("Model loaded. Categories: %s", list(id2cat.values()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception:
        # Don't crash the process on startup failure -- /health will report
        # model_loaded=False so orchestrators can surface it clearly.
        logger.exception("Failed to load model at startup")
    yield


app = FastAPI(title="AI Request Dispatcher", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=state["model"] is not None)


@app.get("/categories", response_model=CategoriesResponse)
def categories():
    if state["id2cat"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return CategoriesResponse(categories=list(state["id2cat"].values()))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokenizer, model, device, id2cat = (
        state["tokenizer"], state["model"], state["device"], state["id2cat"]
    )

    inputs = tokenizer(
        request.text,
        truncation=True,
        max_length=settings.max_seq_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    probabilities = {id2cat[i]: round(float(p), 4) for i, p in enumerate(probs.tolist())}

    return PredictResponse(
        category=id2cat[pred_id],
        confidence=round(float(probs[pred_id].item()), 4),
        probabilities=probabilities,
    )

"""
FastAPI Inference Server for Transformers Models

Features:
- Health checks and readiness probes
- Prometheus metrics
- Request validation with Pydantic
- Async inference with batching support
- Structured logging
"""

import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
model = None
tokenizer = None
classifier = None


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=512, description="Input text for classification"
    )
    return_all_scores: bool = Field(False, description="Return scores for all labels")


class PredictResponse(BaseModel):
    label: str
    score: float
    all_scores: list[dict] | None = None
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, tokenizer, classifier

    logger.info(f"Loading model: {MODEL_NAME}")
    start_time = time.time()

    device = 0 if torch.cuda.is_available() else -1
    if device == -1 and torch.backends.mps.is_available():
        device_str = "mps"
    elif device == -1:
        device_str = "cpu"
    else:
        device_str = "cuda"

    logger.info(f"Using device: {device_str}")

    try:
        classifier = pipeline(
            "sentiment-analysis", model=MODEL_NAME, device=device, tokenizer=MODEL_NAME
        )
        tokenizer = classifier.tokenizer
        model = classifier.model

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down server")
    del model, tokenizer, classifier


app = FastAPI(
    title="Transformers Inference API",
    description="Production-ready inference server for transformer models",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device,
    )


@app.get("/ready")
async def ready():
    """Readiness probe for Kubernetes"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run inference on input text"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        result = classifier(
            request.text,
            return_all_scores=request.return_all_scores,
            truncation=True,
            max_length=512,
        )

        inference_time = (time.time() - start_time) * 1000

        if request.return_all_scores:
            primary_result = max(result[0], key=lambda x: x["score"])
            return PredictResponse(
                label=primary_result["label"],
                score=primary_result["score"],
                all_scores=result[0],
                inference_time_ms=inference_time,
            )
        else:
            return PredictResponse(
                label=result[0]["label"], score=result[0]["score"], inference_time_ms=inference_time
            )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Transformers Inference API",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

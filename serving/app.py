"""
FastAPI Inference Server for Transformers Models

Features:
- Health checks and readiness probes
- Prometheus metrics
- Request validation with Pydantic
- Async inference with batching support
- Structured logging
"""

import asyncio
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
runtime_device: torch.device | None = None


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
    global model, tokenizer, classifier, runtime_device

    logger.info(f"Loading model: {MODEL_NAME}")
    start_time = time.time()

    runtime_device = _select_device()
    device_arg = _device_to_pipeline_arg(runtime_device)
    device_str = runtime_device.type if isinstance(runtime_device, torch.device) else "cuda"

    logger.info(f"Using device: {device_str}")

    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=device_arg,
            tokenizer=MODEL_NAME,
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
    model = None
    tokenizer = None
    classifier = None
    runtime_device = None


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
    device = runtime_device.type if runtime_device is not None else _describe_device()
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
        result = await asyncio.to_thread(
            classifier,
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


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _device_to_pipeline_arg(device: torch.device) -> int | torch.device:
    if device.type == "cuda":
        return 0
    if device.type == "mps":
        return device
    if device.type == "cpu":
        return -1
    return -1


def _describe_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conlist, confloat
from .model import train_model, predict


class Features(BaseModel):
    # Enforce exactly four numeric features (ints allowed, coerced to float).
    values: conlist(confloat(strict=False), min_length=4, max_length=4)  # type: ignore[arg-type]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train a tiny demo model on startup (fast, for demo purposes).
    app.state.model = train_model()
    yield


app = FastAPI(title="mlops-quickstart: demo inference", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # Normalize validation errors to 400 with structured details.
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(feat: Features):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pred, probs = predict(model, feat.values)
    return {"prediction": pred, "probabilities": probs}

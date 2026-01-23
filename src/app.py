from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from .model import train_model, predict
from .logging_util import configure_logging, RequestContextMiddleware


class Features(BaseModel):
    # Enforce exactly four numeric features (ints allowed, coerced to float).
    values: list[float] = Field(min_length=4, max_length=4)

    @field_validator("values")
    @classmethod
    def ensure_numbers(cls, v: list[float]) -> list[float]:
        if any(not isinstance(val, (int, float)) for val in v):
            raise ValueError("values must be numbers")
        return [float(val) for val in v]


def get_model():
    """Centralized dependency to ensure model is loaded."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train a tiny demo model on startup (fast, for demo purposes).
    app.state.model = train_model()
    yield


configure_logging()
app = FastAPI(title="mlops-quickstart: demo inference", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # Normalize validation errors to 400 with structured details.
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


@app.get("/health")
def health(model=Depends(get_model)):
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(feat: Features, model=Depends(get_model)):
    pred, probs = predict(model, feat.values)
    return {"prediction": pred, "probabilities": probs}

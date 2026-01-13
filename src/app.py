from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .model import train_model, predict


class Features(BaseModel):
    values: List[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train a tiny demo model on startup (fast, for demo purposes).
    app.state.model = train_model()
    yield


app = FastAPI(title="mlops-quickstart: demo inference", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(feat: Features):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(feat.values) == 0:
        raise HTTPException(status_code=400, detail="Empty features")
    pred, probs = predict(model, feat.values)
    return {"prediction": pred, "probabilities": probs}

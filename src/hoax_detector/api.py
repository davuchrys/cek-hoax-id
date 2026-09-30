import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .refresh import start_background_refresh

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", "data/corpus.csv"))
STATIC_DIR = Path(__file__).parent / "static"
state: dict = {}


def load_model() -> None:
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    stop = None
    hours = float(os.getenv("AUTO_UPDATE_HOURS", "0"))
    if hours > 0:
        stop = start_background_refresh(hours, CORPUS_PATH, MODEL_PATH, load_model)
    yield
    if stop:
        stop.set()
    state.clear()


app = FastAPI(title="Cek Hoax ID", version=__version__, lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    hoax_probability: float


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/model-info")
def model_info():
    metrics_file = MODEL_PATH.with_name("metrics.json")
    if not metrics_file.exists():
        return {"n_train": None, "n_test": None, "f1": None}
    m = json.loads(metrics_file.read_text())
    return {k: m.get(k) for k in ("n_train", "n_test", "f1")}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "version": __version__}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    prob = float(model.predict_proba([req.text])[0][1])
    return PredictResponse(label="hoax" if prob >= 0.5 else "valid", hoax_probability=round(prob, 4))

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hoax_detector import api
from hoax_detector.preprocess import clean_text
from hoax_detector.train import train

DATA = Path(__file__).parent.parent / "data" / "sample.csv"


def test_clean_text_strips_noise():
    assert clean_text("SEBARKAN!!! https://x.co/abc @user #viral Halo") == "sebarkan halo"


@pytest.fixture()
def client(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics = train(DATA, model_path)
    assert metrics["f1"] > 0.5
    api.MODEL_PATH = model_path
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_hoax_vs_valid(client):
    hoax = client.post("/predict", json={"text": "Sebarkan! Minum ramuan ini sembuhkan kanker, dokter menyembunyikan fakta"})
    valid = client.post("/predict", json={"text": "BMKG mengimbau warga waspada hujan tinggi dan banjir"})
    assert hoax.json()["label"] == "hoax"
    assert valid.json()["label"] == "valid"


def test_predict_rejects_short_text(client):
    assert client.post("/predict", json={"text": "a"}).status_code == 422

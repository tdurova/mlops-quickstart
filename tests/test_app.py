import pytest
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture()
def client():
    # Use TestClient as a context manager to trigger FastAPI lifespan events.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(client):
    # Iris sample (first row)
    sample = [5.1, 3.5, 1.4, 0.2]
    r = client.post('/predict', json={'values': sample})
    assert r.status_code == 200
    data = r.json()
    assert 'prediction' in data
    assert 'probabilities' in data
    assert isinstance(data['probabilities'], list)


def test_predict_empty_values_returns_400(client):
    r = client.post('/predict', json={'values': []})
    assert r.status_code == 400
    data = r.json()
    assert data['detail'] == 'Empty features'


def test_predict_probabilities_shape_and_sum(client):
    sample = [5.1, 3.5, 1.4, 0.2]
    r = client.post('/predict', json={'values': sample})
    assert r.status_code == 200
    data = r.json()
    probs = data['probabilities']
    assert isinstance(probs, list)
    assert len(probs) == 3
    # Sum of probabilities should be ~1.0 (allow small numerical error).
    assert abs(sum(probs) - 1.0) < 1e-6

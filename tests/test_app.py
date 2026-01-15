import asyncio
from contextlib import suppress
import pytest
import httpx
from src.app import app


@pytest.fixture()
def api_request():
    def _request(method: str, path: str, json: dict | None = None):
        async def _call():
            # Keep an additional task alive to avoid event-loop edge cases in this environment.
            sleeper = asyncio.create_task(asyncio.sleep(0.1))
            transport = httpx.ASGITransport(app=app)
            try:
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                        return await c.request(method, path, json=json)
            finally:
                sleeper.cancel()
                with suppress(asyncio.CancelledError):
                    await sleeper

        return asyncio.run(_call())

    return _request


def test_health(api_request):
    r = api_request('GET', '/health')
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(api_request):
    # Iris sample (first row)
    sample = [5.1, 3.5, 1.4, 0.2]
    r = api_request('POST', '/predict', json={'values': sample})
    assert r.status_code == 200
    data = r.json()
    assert 'prediction' in data
    assert 'probabilities' in data
    assert isinstance(data['probabilities'], list)


def test_predict_probabilities_shape_and_sum(api_request):
    sample = [5.1, 3.5, 1.4, 0.2]
    r = api_request('POST', '/predict', json={'values': sample})
    assert r.status_code == 200
    data = r.json()
    probs = data['probabilities']
    assert isinstance(probs, list)
    assert len(probs) == 3
    # Sum of probabilities should be ~1.0 (allow small numerical error).
    assert abs(sum(probs) - 1.0) < 1e-6


def test_predict_wrong_length_returns_400(api_request):
    r = api_request('POST', '/predict', json={'values': [1.0, 2.0, 3.0]})
    assert r.status_code == 400
    data = r.json()
    assert data['detail'] == 'Invalid request'
    assert data['errors']


def test_predict_non_numeric_returns_400(api_request):
    r = api_request('POST', '/predict', json={'values': [1.0, "bad", 3.0, 4.0]})
    assert r.status_code == 400
    data = r.json()
    assert data['detail'] == 'Invalid request'


def test_predict_null_in_payload_returns_400(api_request):
    r = api_request('POST', '/predict', json={'values': [1.0, None, 3.0, 4.0]})
    assert r.status_code == 400
    data = r.json()
    assert data['detail'] == 'Invalid request'

# mlops-quickstart

Minimal MLOps demo: a Dockerized FastAPI inference service with CI.

What’s included
- `src/` – FastAPI app and a tiny demo model trainer
- `Dockerfile` – containerize the inference service
- `.github/workflows/test.yml` – CI that runs `pytest`
- `tests/` – simple pytest tests

Quick start (local)

1. Create a venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
uvicorn src.app:app --reload --port 8080
```

3. Example request:

```bash
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"values": [5.1,3.5,1.4,0.2]}'
```

CI

The workflow `.github/workflows/test.yml` installs requirements and runs tests.

Push & pin (suggested)

```bash
git init
git add .
git commit -m "chore: add mlops-quickstart demo"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/mlops-quickstart.git
git push -u origin main
```

Then pin the repo to your profile and place it in slot #1.

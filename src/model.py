from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_model():
    """Train a tiny demo model (Iris) and return the fitted pipeline."""
    data = load_iris()
    X, y = data.data, data.target
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X, y)
    return pipe


def predict(model, features):
    """Predict class and probabilities for a single sample (list of floats)."""
    pred = model.predict([features])[0]
    probs = model.predict_proba([features])[0].tolist()
    return int(pred), probs

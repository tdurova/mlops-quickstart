from __future__ import annotations

import numpy as np

from src.model import predict, train_model


def test_train_model_returns_sklearn_pipeline() -> None:
    model = train_model()
    steps = dict(model.named_steps)
    assert "scale" in steps
    assert "clf" in steps


def test_predict_returns_int_and_probabilities_sum_to_one() -> None:
    model = train_model()
    sample = np.array([5.1, 3.5, 1.4, 0.2])

    pred, probs = predict(model, sample)

    assert isinstance(pred, int)
    assert isinstance(probs, list)
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6

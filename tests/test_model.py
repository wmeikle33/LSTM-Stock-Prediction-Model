import numpy as np
import pytest

from stock_lstm.model import build_model


def test_build_model_compiles_and_runs_forward_pass():
    model = build_model(window=10, n_features=5)

    X = np.random.rand(4, 10, 5).astype("float32")
    y_pred = model.predict(X, verbose=0)

    assert y_pred.shape == (4, 1)

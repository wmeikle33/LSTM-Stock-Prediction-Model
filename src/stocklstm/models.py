import numpy as np

class NaiveLastValue:
    def fit(self, X: np.ndarray, y: np.ndarray):
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, -1].astype(np.float32)

class LSTMModel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('LSTMModel is private; implement in your internal repo.')

import numpy as np
from .interfaces import Series, Dataset

class Windowizer:
    def __init__(self, lookback: int = 20):
        self.lookback = int(lookback)
    def fit(self, series: Series):
        return self
    def transform(self, series: Series) -> Dataset:
        s = series.values
        L = self.lookback
        if s.ndim != 1 or len(s) <= L:
            raise ValueError('Series must be 1D and longer than lookback.')
        N = len(s) - L
        X = np.zeros((N, L), dtype=np.float32)
        y = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            X[i] = s[i:i+L]
            y[i] = s[i+L]
        return Dataset(X=X, y=y)

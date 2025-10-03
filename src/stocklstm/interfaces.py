from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class Series:
    values: np.ndarray

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray

class Loader(Protocol):
    def load(self) -> Series: ...

class Preprocessor(Protocol):
    def fit(self, series: Series): ...
    def transform(self, series: Series) -> Dataset: ...

class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray): ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

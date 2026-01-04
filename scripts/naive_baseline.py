import numpy as np

def naive_persistence_forecast(y: np.ndarray) -> np.ndarray:
    return y[:-1]

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

y = np.array([100, 101, 102, 101, 103])
pred = naive_persistence_forecast(y)
print(mae(y[1:], pred))  

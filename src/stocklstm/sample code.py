# pip install tensorflow  # if you don't have it
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1) Make a synthetic time series -----------------------------------------
def make_series(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = np.sin(0.03 * t) + 0.3 * np.sin(0.07 * t) + 0.1 * rng.standard_normal(n)
    return y.astype("float32")

def make_windows(series, lookback=20):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    X = np.array(X, dtype="float32")[..., None]  # shape: (N, lookback, 1)
    y = np.array(y, dtype="float32")            # shape: (N,)
    return X, y

series = make_series()
lookback = 20
X, y = make_windows(series, lookback=lookback)

# Train / test split
split = int(0.8 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

# --- 2) Build a simple LSTM model --------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(lookback, 1)),
    layers.LSTM(64),
    layers.Dense(1),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# --- 3) Train & evaluate ------------------------------------------------------
model.fit(X_tr, y_tr, epochs=10, batch_size=64, validation_split=0.1, verbose=2)
loss, mae = model.evaluate(X_te, y_te, verbose=0)
print(f"Test MSE: {loss:.4f} | Test MAE: {mae:.4f}")

# --- 4) Predict a few steps ---------------------------------------------------
pred = model.predict(X_te[:5], verbose=0).squeeze()
print("Pred:", np.round(pred, 3))
print("True:", np.round(y_te[:5], 3))

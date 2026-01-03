# pip install tensorflow  # if you don't have it
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

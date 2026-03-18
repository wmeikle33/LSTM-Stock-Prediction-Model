# src/stock_lstm/model.py
from tensorflow import keras
from tensorflow.keras import layers


def build_model(lookback: int, n_features: int = 1) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(64),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

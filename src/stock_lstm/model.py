from tensorflow import keras


def build_model(window: int, n_features: int, lstm_units: int = 64, dropout: float = 0.2):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window, n_features)),
            keras.layers.LSTM(lstm_units, return_sequences=False),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model

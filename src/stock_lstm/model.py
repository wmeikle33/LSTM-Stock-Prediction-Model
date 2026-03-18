
import keras
from keras import layers

def build_model(window: int, n_features: int, lstm_units: int = 64, dropout: float = 0.2):
    inputs = keras.Input(shape=(window, n_features))
    x = layers.LSTM(lstm_units)(inputs)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="next_close")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

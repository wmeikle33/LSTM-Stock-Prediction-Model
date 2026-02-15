model = keras.Sequential([
    layers.Input(shape=(lookback, 1)),
    layers.LSTM(64),
    layers.Dense(1),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()


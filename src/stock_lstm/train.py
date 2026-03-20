from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import argparse

from stock_lstm.model import build_model


def make_sequences(features: np.ndarray, target: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def train_model(
    csv_path: str,
    outdir: str,
    target_col: str = "close",
    feature_cols=None,
    window: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    target_col = target_col.lower()
    if feature_cols is None:
        feature_cols = ["open", "high", "low", "close", "volume"]
    feature_cols = [c.lower() for c in feature_cols]

    df = df.dropna().sort_values("date").reset_index(drop=True)

    # chronological split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    train_X_raw = train_df[feature_cols].values
    val_X_raw = val_df[feature_cols].values

    train_y_raw = train_df[[target_col]].values
    val_y_raw = val_df[[target_col]].values

    train_X_scaled = x_scaler.fit_transform(train_X_raw)
    val_X_scaled = x_scaler.transform(val_X_raw)

    train_y_scaled = y_scaler.fit_transform(train_y_raw).ravel()
    val_y_scaled = y_scaler.transform(val_y_raw).ravel()

    X_train, y_train = make_sequences(train_X_scaled, train_y_scaled, window)
    X_val, y_val = make_sequences(val_X_scaled, val_y_scaled, window)

    model = build_model(window=window, n_features=len(feature_cols))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(outdir / "model.keras")
    joblib.dump(x_scaler, outdir / "x_scaler.joblib")
    joblib.dump(y_scaler, outdir / "y_scaler.joblib")

    metadata = {
        "window": window,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return history

def main():
    parser = argparse.ArgumentParser(description="Train the LSTM stock model")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--outdir", required=True, help="Directory to save model artifacts")
    parser.add_argument("--target", default="close", help="Target column name")
    parser.add_argument("--window", type=int, default=60, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    train_model(
        csv_path=args.data,
        outdir=args.outdir,
        target_col=args.target,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()

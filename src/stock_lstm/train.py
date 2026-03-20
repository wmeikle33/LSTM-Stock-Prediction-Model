from pathlib import Path
import json
import joblib

from tensorflow import keras

from stock_lstm.baselines import naive_last_close, moving_average
from stock_lstm.metrics import save_eval_artifacts

from stock_lstm.data import (
    load_price_data,
    chronological_split,
    prepare_split_sequences,
)
from stock_lstm.model import build_model


def train_model(
    csv_path: str,
    outdir: str,
    target_col: str = "close",
    feature_cols=None,
    window: int = 60,
    horizon: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    epochs: int = 10,
    batch_size: int = 32,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = ["open", "high", "low", "close", "volume"]

    target_col = target_col.lower()
    feature_cols = [c.lower() for c in feature_cols]

    df = load_price_data(csv_path)
    split = chronological_split(df, train_frac=train_frac, val_frac=val_frac)

    scaler, X_train, y_train, X_val, y_val, X_test, y_test = prepare_split_sequences(
        split=split,
        feature_cols=feature_cols,
        target_col=target_col,
        window=window,
        horizon=horizon,
    )

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

    y_pred_lstm = model.predict(X_test).ravel()
    y_pred_naive = naive_last_close(X_test)
    y_pred_ma = moving_average(X_test)
    results = {
    "naive": evaluate_regression(y_test, y_pred_naive),
    "moving_avg": evaluate_regression(y_test, y_pred_ma),
    "lstm": evaluate_regression(y_test, y_pred_lstm),
    }

    model.save(outdir / "model.keras")
    joblib.dump(scaler, outdir / "x_scaler.joblib")

    metadata = {
        "window": window,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_frac": train_frac,
        "val_frac": val_frac,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "history": history.history,
        "X_test": X_test,
        "y_test": y_test,
    }

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

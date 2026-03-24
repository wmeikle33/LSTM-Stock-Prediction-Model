from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import argparse


def make_inference_sequences(features: np.ndarray, window: int):
    X = []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
    return np.array(X, dtype=np.float32)


def predict_from_csv(model_dir: str, input_csv: str, output_csv: str):
    model_dir = Path(model_dir)

    model = keras.models.load_model(model_dir / "model.keras")
    x_scaler = joblib.load(model_dir / "x_scaler.joblib")
    y_scaler = joblib.load(model_dir / "y_scaler.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())

    window = metadata["window"]
    feature_cols = ["open", "high", "low", "close", "volume"]

    df = pd.read_csv(input_csv)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)

    raw_features = df[feature_cols].values
    scaled_features = x_scaler.transform(raw_features)

    X_pred = make_inference_sequences(scaled_features, window)
    preds_scaled = model.predict(X_pred, verbose=0)
    preds = y_scaler.inverse_transform(preds_scaled).ravel()

    result = df.iloc[window:].copy()
    result["predicted_close"] = preds
    result.to_csv(output_csv, index=False)
    return result

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained LSTM model")
    parser.add_argument("--model", required=True, help="Directory containing model artifacts")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--out", required=True, help="Path to output predictions CSV")
    args = parser.parse_args()

    predict_from_csv(
        model_dir=args.model,
        input_csv=args.data,
        output_csv=args.out,
    )


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pandas as pd


def test_train_then_predict_smoke(tmp_path, sample_prices_df):
    # Write sample CSV
    csv_path = tmp_path / "prices.csv"
    sample_prices_df.to_csv(csv_path, index=False)

    # Import after writing data, so the module path stays simple
    from stock_lstm.train import train_model
    from stock_lstm.predict import predict_from_csv

    outdir = tmp_path / "outputs"
    outdir.mkdir()

    train_model(
        csv_path=str(csv_path),
        outdir=str(outdir),
        target_col="close",
        feature_cols=["open", "high", "low", "close", "volume"],
        window=10,
        epochs=1,
        batch_size=8,
    )

    assert (outdir / "model.keras").exists()
    assert (outdir / "x_scaler.joblib").exists()
    assert (outdir / "y_scaler.joblib").exists()
    assert (outdir / "metadata.json").exists()

    pred_csv = tmp_path / "predictions.csv"
    result = predict_from_csv(
        model_dir=str(outdir),
        input_csv=str(csv_path),
        output_csv=str(pred_csv),
    )

    assert pred_csv.exists()
    assert "predicted_close" in result.columns
    assert len(result) == len(sample_prices_df) - 10

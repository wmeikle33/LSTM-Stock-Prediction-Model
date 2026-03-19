import numpy as np
import pandas as pd
import pytest

from stock_lstm.data import (
    load_price_data,
    chronological_split,
    make_sequences,
    prepare_split_sequences,
)


def test_load_price_data_normalizes_columns_and_sorts(tmp_path):
    csv_path = tmp_path / "prices.csv"

    df = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "Open": [3.0, 1.0, 2.0],
            "High": [3.5, 1.5, 2.5],
            "Low": [2.5, 0.5, 1.5],
            "Close": [3.2, 1.2, 2.2],
            "Volume": [300, 100, 200],
        }
    )
    df.to_csv(csv_path, index=False)

    out = load_price_data(str(csv_path))

    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["date"].is_monotonic_increasing
    assert len(out) == 3


def test_chronological_split_preserves_order(sample_prices_df):
    split = chronological_split(sample_prices_df, train_frac=0.7, val_frac=0.15)

    assert len(split.train_df) == 84
    assert len(split.val_df) == 18
    assert len(split.test_df) == 18

    assert split.train_df["date"].max() < split.val_df["date"].min()
    assert split.val_df["date"].max() < split.test_df["date"].min()


def test_make_sequences_shapes_and_alignment():
    feature_array = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
        dtype=np.float32,
    )
    target_array = np.array([100, 200, 300, 400, 500], dtype=np.float32)

    X, y = make_sequences(feature_array, target_array, window=2, horizon=1)

    assert X.shape == (3, 2, 2)
    assert y.shape == (3,)
    np.testing.assert_array_equal(X[0], np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32))
    assert y[0] == 300.0
    assert y[-1] == 500.0


def test_prepare_split_sequences_returns_usable_arrays(sample_prices_df):
    split = chronological_split(sample_prices_df, train_frac=0.7, val_frac=0.15)

    scaler, X_train, y_train, X_val, y_val, X_test, y_test = prepare_split_sequences(
        split=split,
        feature_cols=["open", "high", "low", "close", "volume"],
        target_col="close",
        window=10,
        horizon=1,
    )

    assert scaler is not None
    assert X_train.ndim == 3
    assert X_val.ndim == 3
    assert X_test.ndim == 3
    assert y_train.ndim == 1
    assert X_train.shape[1:] == (10, 5)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_price_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitData:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Split produced an empty partition; use more data or adjust fractions")

    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: List[str],
) -> np.ndarray:
    return scaler.transform(df[feature_cols].values)


def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    window: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    X, y = [], []
    max_start = len(feature_array) - window - horizon + 1

    for start_idx in range(max_start):
        end_idx = start_idx + window
        target_idx = end_idx + horizon - 1

        X.append(feature_array[start_idx:end_idx])
        y.append(target_array[target_idx])

    if not X:
        raise ValueError("Not enough rows to create sequences")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_split_sequences(
    split: SplitData,
    feature_cols: List[str],
    target_col: str,
    window: int,
    horizon: int = 1,
):
    scaler = fit_feature_scaler(split.train_df, feature_cols)

    X_train_features = transform_features(split.train_df, scaler, feature_cols)
    X_val_features = transform_features(split.val_df, scaler, feature_cols)
    X_test_features = transform_features(split.test_df, scaler, feature_cols)

    y_train_raw = split.train_df[target_col].values.astype(np.float32)
    y_val_raw = split.val_df[target_col].values.astype(np.float32)
    y_test_raw = split.test_df[target_col].values.astype(np.float32)

    X_train, y_train = make_sequences(X_train_features, y_train_raw, window, horizon)
    X_val, y_val = make_sequences(X_val_features, y_val_raw, window, horizon)
    X_test, y_test = make_sequences(X_test_features, y_test_raw, window, horizon)

    return scaler, X_train, y_train, X_val, y_val, X_test, y_test

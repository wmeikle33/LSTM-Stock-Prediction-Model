import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices_df():
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100, 120, n),
            "high": np.linspace(101, 121, n),
            "low": np.linspace(99, 119, n),
            "close": np.linspace(100.5, 120.5, n),
            "volume": np.arange(n) + 1000,
        }
    )
    return df

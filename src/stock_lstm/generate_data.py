import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 300
dates = pd.date_range(start="2020-01-01", periods=n)

returns = np.random.normal(0.0005, 0.02, n)
price = 100 * np.exp(np.cumsum(returns))

df = pd.DataFrame({"date": dates})
df["close"] = price
df["open"] = df["close"].shift(1).fillna(df["close"])
df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.rand(n) * 0.01)
df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.rand(n) * 0.01)
df["volume"] = np.random.randint(1000, 10000, n)

df = df[["date", "open", "high", "low", "close", "volume"]]

out_path = Path("sample_data/prices.csv")
out_path.parent.mkdir(exist_ok=True)

df.to_csv(out_path, index=False)

print(f"Saved synthetic data to {out_path}")

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_data(n, start_date, seed):
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n)

    returns = np.random.normal(0.0005, 0.02, n)
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"date": dates})
    df["close"] = price
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.rand(n) * 0.01)
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.rand(n) * 0.01)
    df["volume"] = np.random.randint(1000, 10000, n)

    return df[["date", "open", "high", "low", "close", "volume"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = generate_data(args.n, args.start_date, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

def train(data_path: str, target: str, window: int, horizon: int, epochs: int, outdir: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Training with:")
    print(f"  data={data_path}")
    print(f"  target={target}")
    print(f"  window={window}, horizon={horizon}, epochs={epochs}")
    print(f"  outdir={outdir}")

def make_run_dir(base: str = "outputs", name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = name or f"run_{ts}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def main():
    p = argparse.ArgumentParser(description="Train LSTM stock prediction model")
    p.add_argument("--data", required=True, help="Path to CSV containing price data")
    p.add_argument("--target", default="Close", help="Target column name (default: Close)")
    p.add_argument("--window", type=int, default=60, help="Lookback window length")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon (steps ahead)")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--outdir", default="outputs", help="Output directory for artifacts")

    args = p.parse_args()
    train(args.data, args.target, args.window, args.horizon, args.epochs, args.outdir)

if __name__ == "__main__":
    main()

import argparse
from stock_lstm.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target", default="close")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
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

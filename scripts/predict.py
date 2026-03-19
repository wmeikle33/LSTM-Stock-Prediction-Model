import argparse
from stock_lstm.predict import predict_from_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predict_from_csv(
        model_dir=args.model_dir,
        input_csv=args.input,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

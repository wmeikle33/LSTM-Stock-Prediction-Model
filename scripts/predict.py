import argparse

from stock_lstm.predict import predict_from_csv

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

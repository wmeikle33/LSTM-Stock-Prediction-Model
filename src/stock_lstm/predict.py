DEFAULT_FEATURE_COLS = ["open", "high", "low", "close", "volume"]


def parse_args():
    ap = argparse.ArgumentParser(description="Predict next closing price from stock history")
    ap.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    ap.add_argument("--scaler", required=True, help="Path to fitted scaler (.joblib)")
    ap.add_argument("--input", required=True, help="CSV containing historical OHLCV data")
    ap.add_argument("--output", default="predictions.csv", help="Where to save predictions")
    ap.add_argument("--window", type=int, default=60, help="Lookback window length")
    ap.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    ap.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    return ap.parse_args()


def build_last_window(df: pd.DataFrame, scaler, feature_cols, window: int) -> torch.Tensor:
    if len(df) < window:
        raise ValueError(
            f"Need at least {window} rows for prediction, but got {len(df)}"
        )

    features = transform_features(df, scaler, feature_cols)
    last_window = features[-window:]  # shape: (window, n_features)
    x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)  # (1, window, n_features)
    return x


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    feature_cols = DEFAULT_FEATURE_COLS

    df = load_price_data(args.input)
    scaler = load(args.scaler)

    model = LSTMRegressor(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    x = build_last_window(df, scaler, feature_cols, args.window).to(device)

    with torch.no_grad():
        pred_close = model(x).squeeze().item()

    last_date = pd.to_datetime(df["date"].iloc[-1])
    out = pd.DataFrame(
        {
            "last_date": [last_date],
            "pred_close": [pred_close],
        }
    )
    out.to_csv(args.output, index=False)
    print(f"Saved prediction to: {args.output}")


if __name__ == "__main__":
    main()

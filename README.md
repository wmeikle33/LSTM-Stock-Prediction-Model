# LSTM Stock Price Prediction

Predicts next-day stock closing prices using an LSTM neural network trained on historical OHLCV data, with comparison against simple baselines.

## Problem

Can a sequence model (LSTM) learn temporal patterns in historical stock data
to predict the next day's closing price more accurately than simple heuristics?

This project frames stock prediction as a supervised time-series regression problem:
- Input: rolling window of past N days of features
- Output: next-day closing price

# LSTM Stock Prediction Model

This repository contains a redaction-safe sample of a stock forecasting pipeline built around an LSTM model for time-series prediction.

The project demonstrates:
- data loading and preprocessing for stock time series
- sequence generation for LSTM inputs
- model training and evaluation
- inference on held-out or new data

It is intended as a public portfolio version of a larger internal project. Proprietary business logic and private datasets have been removed, but the repository preserves the overall workflow and model structure.

## What the model predicts

This sample predicts the next closing price value from a rolling window of historical stock features.

Example framing:
- Input: the previous 30 trading days
- Target: the next trading day's closing price

Depending on how you configure the pipeline, the target can be adapted for:
- next-step regression
- multi-step forecasting
- directional classification

## Input data format

The training and inference scripts expect tabular time-series data in CSV format.

Minimum expected columns:

| column | type | description |
|---|---|---|
| date | string / datetime | trading date |
| open | float | opening price |
| high | float | daily high |
| low | float | daily low |
| close | float | closing price |
| volume | float / int | trading volume |

Example:

```csv
date,open,high,low,close,volume
2024-01-02,185.64,188.44,183.89,187.15,82488700
2024-01-03,187.20,189.10,186.80,188.42,71234000

```

## Quickstart

### 1) Setup
```bash
git clone https://github.com/wmeikle33/LSTM-Stock-Prediction-Model.git
cd LSTM-Stock-Prediction-Model

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -U pip
pip install -e ".[dev]"

python -m scripts.train \
  --data sample_data/prices.csv \
  --target Close \
  --window 60 \
  --horizon 1 \
  --epochs 10 \
  --outdir outputs

python -m scripts.predict \
  --model outputs/model \
  --data sample_data/prices.csv \
  --outdir outputs
```

This repository is a public, redaction‑safe sample of a company program that employs a LSTM based neural network to analyze time series data related to stocks. It demonstrates sample code without exposing any proprietary logic.

✅ You can share this repo publicly. Proprietary parsing rules, vendor/OCR config, and corp data are not included.

## Project structure

```
lstm-stock-prediction-model/
├── src/                   # reusable code (data, features, model)
├── scripts/               # CLI entrypoints: train/predict
├── notebooks/             # original notebook + exported .py
├── sample_data             # place raw data here (gitignored)
├── tests/                 # add unit tests if needed
├── requirements.txt
└── README.md
```

## Disclaimer

This project is for educational and research purposes only.

- It is **not financial advice**.
- It is **not intended for live trading or investment decisions**.
- Historical performance shown in this repository does **not guarantee future results**.
- Any reported performance does **not account for transaction costs, slippage, liquidity constraints, or market impact** unless explicitly stated.

## Baseline

## Evaluation

## Backtesting

## Limitations

Financial markets are complex and inherently unpredictable. Use this code at your own risk.

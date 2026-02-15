
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
├── data/raw/              # place raw data here (gitignored)
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

Financial markets are complex and inherently unpredictable. Use this code at your own risk.

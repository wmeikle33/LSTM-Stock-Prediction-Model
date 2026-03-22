from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import json
import pandas as pd
from pathlib import Path
import numpy as np
from stock_lstm.visualization import plot_actual_vs_pred, plot_residuals

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def regression_metrics(y_true, y_pred, last_close=None) -> dict[str, float]:
    results = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    return results

def save_eval_artifacts(outdir, y_true, y_pred, metrics):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(outdir / "predictions.csv", index=False)

    plot_actual_vs_pred(y_true, y_pred, outdir / "actual_vs_pred.png")
    plot_residuals(y_true, y_pred, outdir / "residuals.png")

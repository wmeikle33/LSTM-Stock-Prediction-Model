import json
import pandas as pd
from pathlib import Path

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

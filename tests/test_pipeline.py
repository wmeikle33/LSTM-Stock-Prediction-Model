import csv, numpy as np
from pathlib import Path
from stocklstm.loaders_csv import CSVCloseLoader
from stocklstm.preprocess import Windowizer
from stocklstm.models import NaiveLastValue
from stocklstm.pipeline import run_train_eval

def write_csv(path: Path, series):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['date','close'])
        base = np.datetime64('2024-01-01')
        for i, v in enumerate(series):
            w.writerow([str(base + np.timedelta64(int(i),'D')), float(v)])

def test_naive_pipeline(tmp_path: Path):
    n = 400
    t = np.arange(n, dtype=float)
    y = np.sin(t*0.05) + 0.05*np.random.default_rng(0).standard_normal(n)
    train, test = y[:250], y[250:]
    tr = tmp_path/'tr.csv'; te = tmp_path/'te.csv'
    write_csv(tr, train); write_csv(te, test)
    metrics = run_train_eval(CSVCloseLoader(str(tr)), CSVCloseLoader(str(te)), Windowizer(lookback=10), NaiveLastValue())
    assert metrics['mse'] < 0.5

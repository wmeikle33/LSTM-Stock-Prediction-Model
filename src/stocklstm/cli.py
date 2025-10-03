import argparse, json
from .loaders_csv import CSVCloseLoader
from .preprocess import Windowizer
from .models import NaiveLastValue
from .pipeline import run_train_eval

def main():
    p = argparse.ArgumentParser(description='Public sample: stock prediction pipeline (NAIVE).')
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--lookback', type=int, default=20)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    metrics = run_train_eval(
        CSVCloseLoader(args.train),
        CSVCloseLoader(args.test),
        Windowizer(lookback=args.lookback),
        NaiveLastValue()
    )
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

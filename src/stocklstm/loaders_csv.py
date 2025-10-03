import csv
import numpy as np
from .interfaces import Series

class CSVCloseLoader:
    def __init__(self, path: str):
        self.path = path
    def load(self) -> Series:
        vals = []
        with open(self.path, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                vals.append(float(row['close']))
        return Series(values=np.asarray(vals, dtype=np.float32))

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    random_state: int = 42
    artifacts_dir: Path = Path("artifacts")
    model_path: Path = artifacts_dir / "model.keras"
    scaler_path: Path = artifacts_dir / "scaler.pkl"
    metrics_path: Path = artifacts_dir / "metrics.json"

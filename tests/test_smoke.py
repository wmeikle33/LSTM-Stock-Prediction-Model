import subprocess
import sys


def run_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_train_module() -> None:
    __import__("stock_lstm.train")


def test_import_predict_module() -> None:
    __import__("stock_lstm.predict")


def test_train_cli_help() -> None:
    result = run_cmd("-m", "stock_lstm.train", "--help")
    assert result.returncode == 0
    assert "Train LSTM stock prediction model" in result.stdout


def test_predict_cli_help() -> None:
    result = run_cmd("-m", "stock_lstm.predict", "--help")
    assert result.returncode == 0
    assert "Predict" in result.stdout

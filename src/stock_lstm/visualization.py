import matplotlib.pyplot as plt

def plot_actual_vs_pred(y_true, y_pred, outpath):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Close")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_residuals(y_true, y_pred, outpath):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.plot(residuals)
    plt.title("Residuals Over Time")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

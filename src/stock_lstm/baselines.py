import numpy as np

def naive_last_close(X_seq, close_idx=3):
    X_seq = np.asarray(X_seq)
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_seq.shape}")
    return X_seq[:, -1, close_idx]

def moving_average(X_seq, close_idx=3):
    X_seq = np.asarray(X_seq)
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_seq.shape}")
    return X_seq[:, :, close_idx].mean(axis=1)

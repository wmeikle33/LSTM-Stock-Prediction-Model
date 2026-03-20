import numpy as np

def naive_last_close(y_prev_close):
    return y_prev_close

def moving_average(close_windows):
    return close_windows.mean(axis=1)

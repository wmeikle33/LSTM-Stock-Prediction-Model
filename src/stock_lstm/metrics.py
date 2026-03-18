from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def metric_score(metric, preds, y_val:
    return metric(preds, y_val)

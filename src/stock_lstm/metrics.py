from sklearn.metrics import accuracy, recall, precision, roc_auc_score

def metric_score(metric, preds, y_val:
    return metric(preds, y_val)

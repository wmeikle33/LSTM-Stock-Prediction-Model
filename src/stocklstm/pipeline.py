from .metrics import mse, mae, mape

def run_train_eval(train_loader, test_loader, prep, model):
    s_tr = train_loader.load()
    s_te = test_loader.load()
    prep.fit(s_tr)
    ds_tr = prep.transform(s_tr)
    ds_te = prep.transform(s_te)
    model.fit(ds_tr.X, ds_tr.y)
    y_pred = model.predict(ds_te.X)
    return {'mse': mse(ds_te.y, y_pred), 'mae': mae(ds_te.y, y_pred), 'mape': mape(ds_te.y, y_pred)}

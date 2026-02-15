split = int(0.8 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

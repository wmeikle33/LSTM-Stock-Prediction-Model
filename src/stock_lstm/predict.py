pred = model.predict(X_te[:5], verbose=0).squeeze()
print("Pred:", np.round(pred, 3))
print("True:", np.round(y_te[:5], 3))

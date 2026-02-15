model.fit(X_tr, y_tr, epochs=10, batch_size=64, validation_split=0.1, verbose=2)
loss, mae = model.evaluate(X_te, y_te, verbose=0)
print(f"Test MSE: {loss:.4f} | Test MAE: {mae:.4f}")

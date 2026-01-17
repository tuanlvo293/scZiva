import numpy as np

def log_mse(X, X_imputed):
    log_mse = np.mean((np.log1p(X_imputed) - np.log1p(X)) ** 2)
    return log_mse.round(5)

def mae(X, X_imputed):
    mae = np.mean(np.abs(X_imputed - X))
    return mae.round(5)

def rmse(X, X_imputed):
    rmse = np.sqrt(np.mean((X_imputed - X) ** 2))
    return rmse.round(5)

def pcc(X, Y):
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    num = np.mean(X*Y) - np.mean(X)*np.mean(Y)
    den = np.sqrt(np.mean(X**2) - np.mean(X)**2) * np.sqrt(np.mean(Y**2) - np.mean(Y)**2)
    return (num / den).round(5) if den != 0 else 0.0



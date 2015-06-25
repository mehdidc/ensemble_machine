import numpy as np

def logloss(pred, y):
    probs = pred[np.arange(pred.shape[0]), y]
    probs = np.maximum(np.minimum(probs, 1 - 1e-15), 1e-15)
    return -np.mean(np.log(probs))

import numpy as np

def get_lossFunction(lossFunction_name: str):
    lossFunctions = {
        'mse': MSE,
    }
    # Returns the loss function method; defaults to MSE if not found
    return lossFunctions.get(lossFunction_name.lower(), MSE)

def MSE(predictions, targets):
    # Computes the MSE and the gradient of MSE with respect to predictions
    mse = np.mean((predictions - targets) ** 2)
    gradient = 2 * (predictions - targets) / predictions.shape[0]

    return mse, gradient

def CrossEntropy(predictions, targets):
    # Computes the Cross-Entropy and the gradient of Cross-Entropy with respect to predictions
    ce = -np.sum(targets * np.log(predictions)) / predictions.shape[0]
    gradient = -targets / predictions

    return ce, gradient
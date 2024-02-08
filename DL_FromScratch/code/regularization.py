import numpy as np

def get_regularization(regularization_name: str):
    regularization = {
            'none': none,
            'l1': L1,
            'l2': L2
        }
    # Returns the loss function method; defaults to MSE if not found
    return regularization.get(regularization_name.lower(), none)

class L1:
    @staticmethod
    def func(regLambda: float, x: np.ndarray) -> np.ndarray:
        return regLambda * np.sum(np.abs(x))

    @staticmethod
    def grad(regLambda: float, x: np.ndarray) -> np.ndarray:
        return regLambda * np.sign(x)

class L2:
    @staticmethod
    def func(regLambda: float, x: np.ndarray) -> np.ndarray:
        return regLambda * np.sum(x ** 2) / 2

    @staticmethod
    def grad(regLambda: float, x: np.ndarray) -> np.ndarray:
        return regLambda * x

class none:
    @staticmethod
    def func(regLambda: float, x: np.ndarray) -> np.ndarray:
        return 0

    @staticmethod
    def grad(regLambda: float, x: np.ndarray) -> np.ndarray:
        return 0
import numpy as np

def get_activation(activation_name: str):
    activations = {
        'relu': Relu,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'linear': Linear
    }
    # Returns the activation class; defaults to Linear if not found
    return activations.get(activation_name.lower(), Linear)

class Relu:
    @staticmethod
    def func(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, 0, 1)

class Sigmoid:
    @staticmethod
    def func(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        return Sigmoid.func(x) * (1 - Sigmoid.func(x))

class Tanh:
    @staticmethod
    def func(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        return 1 - Tanh.func(x) ** 2

class Softmax:
    @staticmethod
    def func(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        return Softmax.func(x) * (1 - Softmax.func(x))

class Linear:
    @staticmethod
    def func(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    

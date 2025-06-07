import numpy as np

class Activations():

    @staticmethod
    def relu(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z) 
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

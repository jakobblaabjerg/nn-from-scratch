import numpy as np
import copy

from src.activations.functions import Activations
from src.loss.functions import LossFuncs

class FullyConnectedNet():
   
    def __init__(self, n_layers, n_hidden, n_input, n_output):

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_input = n_input

        self.weights = {}
        self.biases = {}

        layer_sizes = [n_input] + [self.n_hidden] * self.n_layers + [self.n_output]

        for i in range(len(layer_sizes) - 1):

            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]

            self.weights[f'W{i+1}'] = np.random.randn(out_size, in_size)
            self.biases[f'b{i+1}'] = np.zeros((out_size, 1))

        self.Z = {}
        self.A = {}

        # derivatives
        self.dZ = {}
        self.dA = {}
        self.dW = {}
        self.db = {}


    def forward_pass(self, X):

        A_prev = X
        self.A['A0'] = A_prev

        total_layers = self.n_layers+1

        for i in range(total_layers):

            W = self.weights[f'W{i+1}']
            b = self.biases[f'b{i+1}']

            Z = np.matmul(W, A_prev)+b
            self.Z[f'Z{i+1}'] = Z

            if i+1 < total_layers:
                A = Activations.relu(Z)
            else:
                A = Activations.softmax(Z)

            self.A[f'A{i+1}'] = A
            A_prev = A


    def get_parameters(self):
        return copy.deepcopy(self.weights), copy.deepcopy(self.biases), copy.deepcopy(self.Z), copy.deepcopy(self.A)





    def train_net(self, X):

        self.init_net(X)

        #forward_pass()



        pass

    def back_prop(self, y):

        dz4 = A4-y
        dw4 = dz4*A3
        db4 = 1*dz4
        da3 = w4*dz4
        dz3 = da3* (z3>0)







        pass








if __name__ == "__main__":
    
    X = np.array([[0.24], [0.31]])
    y = np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])
    net = FullyConnectedNet(n_layers=3, n_hidden=3, n_input=2, n_output=10)
    net.forward_pass(X)
    params = net.get_parameters()
    y_hat = params[-1]['A4']
    loss = LossFuncs.categorical_cross_entropy(y_hat, y)
    print(loss)
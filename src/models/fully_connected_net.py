import numpy as np
import copy

from utils.activations import Activations

class FullyConnectedNet():
   
    def __init__(self, n_layers, n_hidden):

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        self.weights = {}
        self.biases = {}
        self.Z = {}
        self.A = {}


    def init_net(self, X):

        n_input = X.shape[0] 
        n_output = 10
        layer_sizes = [n_input] + [self.n_hidden] * self.n_layers + [n_output]

        for i in range(len(layer_sizes) - 1):

            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]

            self.weights[f'W{i+1}'] = np.random.randn(out_size, in_size)
            self.biases[f'b{i+1}'] = np.zeros((out_size, 1))


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

    def train_net(self):
        pass

    def back_prop():
        pass


if __name__ == "__main__":
    
    X = np.array([[0.24], [0.31]])
    net = FullyConnectedNet(n_layers=3, n_hidden=3)
    net.init_net(X)
    net.forward_pass(X)
    params = net.get_parameters()
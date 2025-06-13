import numpy as np
import copy

from src.activations.functions import Activations
from src.loss.functions import LossFuncs

class FullyConnectedNet():
   
    def __init__(self, n_hidden_layers, n_hidden_units, n_input, n_output):

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.n_output = n_output
        self.n_input = n_input
        self.n_layers = self.n_hidden_layers+2

        self.weights = {}
        self.biases = {}

        layer_sizes = [n_input] + [self.n_hidden_units] * self.n_hidden_layers + [self.n_output]

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

        

        for i in range(self.n_layers-1):

            W = self.weights[f'W{i+1}']
            b = self.biases[f'b{i+1}']

            Z = np.matmul(W, A_prev)+b
            self.Z[f'Z{i+1}'] = Z

            if i+1 < self.n_layers-1:
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

        m = y.shape[1]
        dZ_next = self.A[f"A{self.n_layers-1}"]-y
        self.dZ[f"dZ{self.n_layers-1}"] = dZ_next

        for i in reversed(range(self.n_layers-1)):
          
            W_next = self.W[f"W{i+1}"]
            dA = np.matmul(W_next.T, dZ_next)           
            dZ = dA*(self.Z[f"Z{i}"]>0) # no matrix multiplication, because we did an elementwise derivative

            dW_next = np.matmul(dZ_next, self.A[f"A{i}"].T)/m
            db_next = np.sum(dZ_next, axis=1, keepdims=True)/m

            # self.dA[f"dA{i}"] = dA
            # self.dZ[f"dZ{i}"] = dZ
            self.dW[f"dW{i+1}"] = dW_next
            self.db[f"dB{i+1}"] = db_next
            dZ_next = dZ








if __name__ == "__main__":
    
    X = np.array([[0.24], [0.31]])
    y = np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])
    net = FullyConnectedNet(n_layers=3, n_hidden=3, n_input=2, n_output=10)
    net.forward_pass(X)
    params = net.get_parameters()
    y_hat = params[-1]['A4']
    loss = LossFuncs.categorical_cross_entropy(y_hat, y)
    print(loss)







        #     dZ4 = A4-y
        # dW4 = dZ4*A3
        # db4 = 1*dZ4
        # dA3 = W4*dZ4
        # dZ3 = dA3 * (Z3>0)
        # dW3 = dZ3 * A2 
        # db3 = dZ3 *1
        # dA2 = W3 * dZ3
        # dZ2 = dA2 * (Z2>0)
        # dW2 = dZ2*A1
        # db2 = dZ2 * 1
        # dA1 = W2 * dZ2
        # dZ1 = dA1* (Z1 > 0)
        # dW1 = dZ1 * X
        # db1 = dZ1 * 1
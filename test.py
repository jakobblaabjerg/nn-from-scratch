import numpy as np
from src.models.fully_connected_net import FullyConnectedNet
from src.loss import LossFuncs



# create dummy input and label
X = np.array([[0.24], [0.31]])  # shape: input_dim x batch_size
y = np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]])  # one-hot label, 10 classes

# initialize network
net = FullyConnectedNet(n_layers=3, n_hidden=3, n_input=2, n_output=10)

# run forward pass
net.forward_pass(X)

# get predictions
_, _, _, A = net.get_parameters()
y_hat = A[f"A{net.n_layers + 1}"]

# compute loss
loss = LossFuncs.categorical_cross_entropy(y_hat, y)
print(f"Loss: {loss:.4f}")
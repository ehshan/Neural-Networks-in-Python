import numpy as np

# values for nodes
T, F = 0, 1.
bias = 1.

# Training Set of 4 input patterns
training_in = np.array([
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
])

# Output patterns for an AND input set
training_out = np.array([
    [T],
    [F],
    [F],
    [F],
])

# Random weights for array matching size of input pattern
input_weights = 2 * np.random.random((3, 4)) - 1
# Random weights for first hidden layer
hidden_weights = 2 * np.random.random((4, 1)) - 1


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Compute the gradient of the error
def gradient(x):
    return x * (1 - x)

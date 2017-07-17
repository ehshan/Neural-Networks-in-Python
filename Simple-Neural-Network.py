import numpy as np

# values for nodes
T, F = 0, 1.
bias = 1.

# Training Set of 4 inputs
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

# Generates random weights between -1 & 1 for array matching size of input pattern
weights = 2 * np.random.random((3, 1)) - 1


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x):
    return x * (1 - x)


# 1000 epochs to train on
for epoch in range(10000):
    # Forward propagation - X axis is weight Y axis is error
    # Minimization (error) problem
    input_layer = training_in

    # Applied sigmoid function to product of input and weights
    output_layer = sigmoid(np.dot(input_layer, weights))
    # print("Hidden: ", hidden_layer, "\n")  # values in hidden layer

    # Error is the training out - output
    error = training_out - output_layer
    # print("Raw Error: ", "\n", error, "\n")  # the raw error

    # Computes the mean of error across dimensions of array
    print("epoch: ", epoch, "Error: ", str(np.mean(np.abs(error))), "\n")

    # Product of the error the slope at (X,Y)
    error_gradient = error * gradient(output_layer)

    # Update the weights
    weights += np.dot(input_layer.T, error_gradient)

print("Final Output:", "\n", output_layer)

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


# 10000 epochs
for epoch in range(10000):
    # Input layer
    input_layer = training_in

    # First hidden layer
    hidden_layer_1 = sigmoid(np.dot(input_layer, input_weights))

    # Second hidden layer
    hidden_layer_2 = sigmoid(np.dot(hidden_layer_1, hidden_weights))

    # Calculate the error
    layer_2_error = training_out - hidden_layer_2

    # Computes the mean of elements across dimensions of array
    print("epoch: ", epoch, "Error: ", str(np.mean(np.abs(layer_2_error))), "\n")

    # The plot of the weight(x) value against the error(y)
    layer_2_error_gradient = layer_2_error * gradient(hidden_layer_2)

    # Calculate the error in the first error
    layer_1_error = layer_2_error_gradient.dot(hidden_weights.T)

    # Compute the error gradient in the first layer
    layer_1_error_gradient = layer_1_error * gradient(hidden_layer_1)

    # Update the weights (= x = x + gradient) for both layers
    hidden_weights += np.dot(hidden_layer_1.T, layer_2_error_gradient)
    input_weights += np.dot(input_layer.T, layer_1_error_gradient)

print("Final Output:", "\n", hidden_layer_2)

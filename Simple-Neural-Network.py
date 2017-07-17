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

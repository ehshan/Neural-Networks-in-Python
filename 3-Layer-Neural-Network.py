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

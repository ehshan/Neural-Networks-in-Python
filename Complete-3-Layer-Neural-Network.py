import numpy as np
from sklearn.model_selection import train_test_split

# generate sample data
np.random.seed(1)
data_points = 1000

# STATS FOR 3 CLUSTERS

# cluster means
c1_mean = [0, 0]
c2_mean = [1, 4]
c3_mean = [2, 8]
# cluster co-variance
c1_cov = [[2, .7], [.7, 2]]
c2_cov = [[1, .7], [.7, 1]]
c3_cov = [[0, .7], [.7, 0]]

# generate
# Cluster 1
c1 = np.random.multivariate_normal(c1_mean, c1_cov, data_points)
# Cluster 2
c2 = np.random.multivariate_normal(c2_mean, c2_cov, data_points)
# Cluster 3
c3 = np.random.multivariate_normal(c3_mean, c3_cov, data_points)

# All holding all data for 3 clusters
data_features = np.vstack((c1, c2, c3)).astype(np.float32)

# Even distribution of labels -> 0, 1, 2
data_labels = np.hstack((np.zeros(data_points), np.ones(data_points), np.ones(data_points) + 1))

# One-hot encoding for data labels
onehot_labels = np.zeros((data_labels.shape[0], 3)).astype(int)
onehot_labels[np.arange(len(data_labels)), data_labels.astype(int)] = 1

# split data to train/test
training_data, test_data, training_labels, test_labels = \
    train_test_split(data_features, onehot_labels, test_size=.1, random_state=12)


# Rectifier Activation function
# in <0 -> out=0
# in >0 -> out=in
def relu_activation(x):
    return np.maximum(x, 0)


# normalise output array to probabilities - so sum of values=1
def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)


# Loss function - takes probability array from softmax function & the target label
def cross_entropy_loss(prob_array, target_label):
    i = np.argmax(target_label, axis=1).astype(int)
    predicted_probability = prob_array[np.arange(len(prob_array)), i]
    log_predicted = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_predicted / len(log_predicted))
    return loss


# Regularisation loss
def l2_regularisation(l2_lambda, weight_1, weight_2):
    weight_1_loss = 0.5 * l2_lambda * np.sum(weight_1 * weight_1)
    weight_2_loss = 0.5 * l2_lambda * np.sum(weight_2 * weight_2)
    return weight_1_loss + weight_2_loss


# Assign Network Variables
hidden_nodes = 5
num_labels = training_labels.shape[1]
num_features = training_data.shape[1]
learning_rate = .01
reg_lambda = .01

# Layer 1 array matches number of hidden nodes
layer1_weights = np.random.normal(0, 1, [num_features, hidden_nodes])
# Layer 2 array matches number of labels
layer2_weights = np.random.normal(0, 1, [hidden_nodes, num_labels])

# Biases for layers
layer1_biases = np.zeros((1, hidden_nodes))
layer2_biases = np.zeros((1, num_labels))

# 10000 epochs
for epoch in range(10000):
    # NETWORK STRUCTURE
    # Input layer
    input_layer = np.dot(training_data, layer1_weights)
    # Hidden Layer
    hidden_layer = relu_activation(input_layer + layer1_biases)
    # Output layer
    output_layer = np.dot(hidden_layer, layer2_weights) + layer2_biases
    # Output pattern
    output_prob = softmax(output_layer)

    # Loss between output and labels
    loss = cross_entropy_loss(output_prob, training_labels)
    loss += l2_regularisation(reg_lambda, layer1_weights, layer2_weights)

    # Calculate the error with network output
    # Divide the average of each individual label loss by number of observations
    output_error = (output_prob - training_labels) / output_prob.shape[0]

    # Calculate the error in hidden layer
    hidden_error = np.dot(output_error, layer2_weights.T)
    # Compensate for ReLu activation function
    hidden_error[hidden_layer <= 0] = 0

    # Error gradient on layer 2 weights and biases
    layer2_weight_gradient = np.dot(hidden_layer.T, output_error)
    layer2_bias_gradient = np.sum(output_error, axis=0, keepdims=True)

    # Error gradient on layer 1 weights and biases
    layer1_weight_gradient = np.dot(training_data.T, hidden_error)
    layer1_bias_gradient = np.sum(hidden_error, axis=0, keepdims=True)

    # Add regularisation to weight gradients
    layer2_weight_gradient += reg_lambda * layer2_weights
    layer1_weight_gradient += reg_lambda * layer1_weights

    # UPDATES THE WEIGHTS AND BIASES
    layer1_weights -= learning_rate * layer1_weight_gradient
    layer1_biases -= learning_rate * layer1_bias_gradient
    layer2_weights -= learning_rate * layer2_weight_gradient
    layer2_biases -= learning_rate * layer2_bias_gradient

    # Computes the loss & mean error for every 1000 epochs
    if (epoch % 1000) == 0:
        print("epoch:", epoch, "Error:", str(np.mean(np.abs(output_error))), "Loss:", loss)


# For labels compute the no correct prediction over total predictions
def accuracy(predictions, labels):
    prediction_boolean = np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(prediction_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy

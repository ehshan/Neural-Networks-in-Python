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


# normalise output array so sum of values=1
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


# Regularization loss
def l2_regularisation(l2_lambda, weight_1, weight_2):
    weight_1_loss = 0.5 * l2_lambda * np.sum(weight_1 * weight_1)
    weight_2_loss = 0.5 * l2_lambda * np.sum(weight_2 * weight_2)
    return weight_1_loss + weight_2_loss

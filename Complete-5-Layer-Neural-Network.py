import numpy as np

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
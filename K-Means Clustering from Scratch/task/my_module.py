import numpy as np


class CustomKMeans:
    def __init__(self, features: np.ndarray, target: np.ndarray):
        self.features = features
        self.target = target
        self.centroids = self.features[0:3]

    def euclidean_distance_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise Euclidean distances between each row of `a` and each row of `b`.
        `a` shape: (n_samples, n_features)
        `b` shape: (n_centroids, n_features)
        Returns a distance matrix of shape (n_samples, n_centroids)
        """
        # Broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
        a_sq = np.sum(a ** 2, axis=1).reshape(-1, 1)  # shape: (n_samples, 1)
        b_sq = np.sum(b ** 2, axis=1).reshape(1, -1)  # shape: (1, n_centroids)
        ab = np.dot(a, b.T)  # shape: (n_samples, n_centroids)
        squared_dists = a_sq - 2 * ab + b_sq
        dists = np.sqrt(np.maximum(squared_dists, 0.0))  # shape: (n_samples, n_centroids)
        return dists

    def find_nearest_center(self) -> np.ndarray:
        """
        Returns the index of the nearest centroid for each feature vector.
        """
        dists = self.euclidean_distance_matrix(self.features, self.centroids)
        return np.argmin(dists, axis=1)  # shape: (n_samples,)
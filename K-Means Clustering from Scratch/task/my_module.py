import numpy as np
from sklearn.metrics import silhouette_score


class CustomKMeans:
    def __init__(self, k: int = 2) -> None:
        """
        Initializes the CustomKMeans clustering model.

        Parameters:
        - k (int): Number of clusters to form. Default is 2.
        """
        self.features = None
        self.clusters_num = k
        self.labels = None
        self.centroids = None
        self.new_centroids = None

    def euclidean_distance_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise Euclidean distance matrix between rows of `a` and `b`.

        Parameters:
        - a (np.ndarray): Array of shape (n_samples, n_features).
        - b (np.ndarray): Array of shape (n_centroids, n_features).

        Returns:
        - np.ndarray: Distance matrix of shape (n_samples, n_centroids).
        """
        # Broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
        a_sq = np.sum(a ** 2, axis=1).reshape(-1, 1)  # shape: (n_samples, 1)
        b_sq = np.sum(b ** 2, axis=1).reshape(1, -1)  # shape: (1, n_centroids)
        ab = np.dot(a, b.T)  # shape: (n_samples, n_centroids)
        squared_dists = a_sq - 2 * ab + b_sq
        dists = np.sqrt(np.maximum(squared_dists, 0.0))  # shape: (n_samples, n_centroids)
        return dists

    def find_nearest_center(self, features=None) -> np.ndarray:
        """
        Assigns each sample to the nearest centroid.

        Parameters:
        - features (np.ndarray, optional): Data to predict labels for. If None, uses self.features.

        Returns:
        - np.ndarray: Array of shape (n_samples,) with cluster labels.
        """
        if features is None:
            features = self.features
        dists = self.euclidean_distance_matrix(features, self.centroids)
        labels = np.argmin(dists, axis=1)  # shape: (n_samples,)
        return labels

    def calculate_new_centers(self) -> None:
        """
        Updates centroids by calculating the mean of all samples assigned to each cluster.
        """
        cluster_labels = np.unique(self.labels)
        new_centroids = []
        for label in cluster_labels:
            centroid = np.mean(self.features[self.labels == label], axis=0)
            new_centroids.append(centroid)
        self.new_centroids = np.array(new_centroids)

    def fit(self, X: np.ndarray, eps: float=1e-6) -> None:
        """
        Computes k-means clustering.

        Parameters:
        - X (np.ndarray): Training instances to cluster.
        - eps (float): Threshold to determine convergence (based on centroid movement).
        """
        self.features = X
        self.new_centroids = None
        self.centroids = self.features[0:self.clusters_num]  # set initial centroids
        self.labels = self.find_nearest_center()  # find initial labels based on initial centroids

        while True:
            self.calculate_new_centers()  # calculate new centroids
            diff = np.linalg.norm(self.centroids - self.new_centroids, axis=1)
            if np.all(diff < eps):
                break
            self.centroids = self.new_centroids
            self.labels = self.find_nearest_center()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the closest cluster each sample in X belongs to.

        Parameters:
        - X (np.ndarray): New data to predict.

        Returns:
        - np.ndarray: Cluster index for each sample in X.
        """
        labels = self.find_nearest_center(features=X)
        return labels

    def compute_inertia(self) -> float:
        """
        Calculates the inertia of the current clustering.

        Inertia is the sum of squared Euclidean distances between each sample
        and the centroid of the cluster it belongs to. It reflects how internally
        coherent the clusters are (lower is better).

        Returns:
            float: The total inertia of the current clustering.
        """
        return np.sum(np.linalg.norm(self.features - self.centroids[self.labels], axis=1) ** 2)

    def compute_silhouette_score(self) -> float:
        """
        Computes the silhouette score for the current clustering.

        The silhouette score measures how similar an object is to its own cluster
        compared to other clusters. The score ranges from -1 to 1:
        - 1 indicates that the sample is far away from the neighboring clusters.
        - 0 indicates that the sample is on or very close to the decision boundary between two clusters.
        - Negative values indicate that samples might have been assigned to the wrong cluster.

        Returns:
            float: The silhouette score of the current clustering.
        """
        return silhouette_score(self.features, self.labels)

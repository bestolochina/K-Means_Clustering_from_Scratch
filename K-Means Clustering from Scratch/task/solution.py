from typing import Any

import numpy as np
from numpy import floating
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# scroll down to the bottom to implement your solution


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):
    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


def euclidean_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    dists = np.sqrt(a_sq - 2 * ab + b_sq)  # shape: (n_samples, n_centroids)
    return dists


def find_nearest_center(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Returns the index of the nearest centroid for each feature vector.
    """
    dists = euclidean_distance_matrix(features, centroids)
    return np.argmin(dists, axis=1)  # shape: (n_samples,)

if __name__ == '__main__':
    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # write your code here
    centroids = X_full[0:3]
    labels = find_nearest_center(X_full[-10:], centroids)
    print(labels.tolist())

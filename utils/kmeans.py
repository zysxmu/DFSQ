import random

import numpy as np
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, n_clusters: int, iterations=100, eps=1e-3):
        self.n_clusters, self.iterations, self.eps, self.centers = n_clusters, iterations, eps, None

    def fit(self, X: np.ndarray):
        self.centers = np.vstack(
            (X[random.sample(range(len(X)), self.n_clusters-1),:], np.zeros((1, X.shape[1]))))

        for _ in range(self.iterations):
            y_pred = self(X)
            centers = np.stack([
                np.mean(X[y_pred == i], axis=0) if np.any(y_pred == i) else random.choice(X)
                for i in range(self.n_clusters - 1)
            ])
            centers = np.vstack((centers, np.zeros((1, X.shape[1]))))

            if np.abs(self.centers - centers).max() < self.eps:
                break

            self.centers = centers

    def __call__(self, X: np.ndarray):
        return np.array([np.argmin(np.linalg.norm(self.centers - x, axis=1)) for x in X])  


def load_data(n_samples_per_class=200, n_classes=5):
    X = np.concatenate([np.random.randn(n_samples_per_class, 2) + 3 * np.random.randn(2) for _ in range(n_classes)])
    y = np.concatenate([np.full(n_samples_per_class, label) for label in range(n_classes)])
    return X, y



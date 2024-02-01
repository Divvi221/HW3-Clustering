import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        scores = np.zeros(X.shape[0])
        dist = cdist(X,X)
        for i in range(len(scores)):
            cluster = y[i]
            other_c = y[y != y[i]]
            a_i = np.mean(dist[i, y == cluster]) if np.sum(y == cluster) > 1 else 0
            b_i = np.min([np.mean(dist[i, y == c]) for c in other_c])
            scores[i] = (b_i - a_i) / max(a_i, b_i) if a_i > 0 or b_i > 0 else 0

        return scores

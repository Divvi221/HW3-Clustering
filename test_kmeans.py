# Write your k-means unit tests here
import numpy as np
from sklearn.cluster import KMeans as skckm
from cluster import (
    KMeans,
    Silhouette
)
import pytest

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

def test_kmeans():
    try:
        k=2
        km = KMeans(k=2)
        km.fit(X)
        km_clusters = km.predict(X)
        centroids = km.get_centroids()
        error = km.get_error()
        assert km_clusters is not None
        assert centroids is not None
        assert error is not None
        cluster_assignments = km.assignments
        cluster_counts = np.bincount(cluster_assignments)
        assert len(cluster_assignments) == X.shape[0]
        assert len(cluster_counts) == k
    except AssertionError as e:
        print(e)
    except Exception as e:
        print(f"Unexpected error:{e}")
    #passing an empty array
    km1 = KMeans(k=2)
    empty = np.array([]).reshape(0, 2)
    try:
        km1.fit(empty)
        print("Fail: expected an exception here since the array is empty")
    except ValueError as e:
        print("Pass: expected a ValueError")
    except Exception as e:
        print("Fail: unexpected Exception")

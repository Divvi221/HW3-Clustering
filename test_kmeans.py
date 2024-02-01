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

def test_fit():
    km1 = KMeans(k=2)
    km2 = skckm(n_clusters=2)
    km1.fit(X)
    km2.fit(X)
    km_cent = np.sort(km1.get_centroids(), axis=0)
    skckm_cent = np.sort(km2.cluster_centers_, axis=0)
    print(skckm_cent)
    np.testing.assert_almost_equal(km_cent, skckm_cent, decimal=1,
                               err_msg="The centroids do not match.")

test_fit()
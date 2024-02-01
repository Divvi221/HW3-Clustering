# write your silhouette score unit tests here
import numpy as np
from sklearn.cluster import KMeans as skckm
from sklearn.metrics import silhouette_samples
from cluster import (
    KMeans,
    Silhouette
)
import pytest

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

def test_silhouette():
    k=2
    km = KMeans(k=2)
    km.fit(X)
    km_clusters = km.predict(X)
    score = Silhouette().score(X,km_clusters)
    km1 = skckm(n_clusters=2)
    score1 = silhouette_samples(X,km1.fit_predict(X), metric='euclidean')
    #print([score.mean(),score1.mean()])
    assert score is score1
    assert score is not None

    #passing an empty array
    km1 = KMeans(k=2)
    empty = np.array([]).reshape(0, 2)
    km1.fit(X)
    km1_clusters = km.predict(X)
    score1 = Silhouette().score(X,km1_clusters)
    try:
        score1 = Silhouette().score(X,km1_clusters)
        print("Fail: expected an exception here since the array is empty")
    except ValueError as e:
        print("Pass: expected a ValueError")
    except Exception as e:
        print("Fail: unexpected Exception")

test_silhouette()
    


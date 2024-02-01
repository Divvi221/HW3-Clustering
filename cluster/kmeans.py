import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if isinstance(k,int)==False or k <= 0:
            raise ValueError("k must be a positive integer")
        if isinstance(tol,float)==False or tol <= 0:
            raise ValueError("tol must be a positive float")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids =  None
        self.data = None
        self.assignments = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not isinstance(mat, np.ndarray) or len(mat.shape) != 2:
            raise ValueError("mat must be a 2D numpy array")
        
        sample_num, feature_num = mat.shape
        rng = np.random.default_rng(12345)
        r_int = rng.choice(sample_num, size = self.k,replace=False) #pick k points out of total sample numbers without replacement
        self.centroids = mat[r_int] #initial centroids
        #print(self.centroids)
        for i in range(self.max_iter):
            distances = cdist(mat, self.centroids,'euclidean')
            #print(distances)
            nearest_c = np.argmin(distances, axis=1) #find min 
            print(nearest_c)
            new_c = np.array([mat[nearest_c == k].mean(axis=0) for k in range(self.k)])
            if np.allclose(self.centroids, new_c, atol=self.tol):
                break
            self.centroids = new_c
            self.assignments = nearest_c
        self.data = mat

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is None:
            raise ValueError("Call the fit function with the appropriate data before calling the predict function.")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("mat.shape[1] and centroids.shape[1] should be equal")
        dist = cdist(mat, self.centroids)
        return np.argmin(dist, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.centroids is None:
            raise ValueError("Call the fit function with the appropriate data before calling the get_error functions")
        dist = cdist(self.data, self.centroids)
        assigned_distances = dist[np.arange(len(self.assignments)), self.assignments]
        squared_distances = assigned_distances ** 2
        mean_squared_error = np.mean(squared_distances)
        return mean_squared_error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("The model has not been fit yet.")
        return self.centroids

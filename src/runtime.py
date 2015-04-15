import CosineCluster as CC
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import time
import numpy as np
import sklearn.cluster as cluster
from numpy.random import normal
from numpy.linalg import norm
import cProfile
import time

def Kmeans_SVD(matrix, K, k):
    (u, s, v) = svds(csr_matrix(matrix, dtype=float), k=k)
    matrix = matrix.dot(v.transpose())
    db = cluster.KMeans(n_clusters=K, max_iter=300)
    return db.fit_predict(matrix)


def Kmeans_JLT(matrix, K, k):
    (n, d) = matrix.shape
    phi = normal(size=(d, k))
    nphi = phi / norm(phi, axis=0)[np.newaxis, :]
    matrix = matrix.dot(nphi)
    db = cluster.KMeans(n_clusters=K, max_iter=300)
    return db.fit_predict(matrix)


def average_sim(labels, matrix):
    total_sim = 0
    for label in np.unique(labels):
        total_sim += CC.cluster_sim(matrix[labels == label, :])
    return total_sim / matrix.shape[0]


if __name__ == "__main__":
    (matrix, subreddit_map, reverse_map) = CC.matrix_repre("../data/subreddits")
    matrix = matrix[np.ravel(matrix.sum(axis=1) > 0), :]
    matrix = CC.normalize_rows(matrix)
    bands = [6] * 30    
    def test_JLT(k):
        labels = Kmeans_JLT(matrix, 100, k)
        print average_sim(labels, matrix)

    def test_SVD(k):
        labels = Kmeans_SVD(matrix, 100, k)
        print average_sim(labels, matrix)

    def test_LSH(bands):
        #labels = Kmeans_SVD(matrix, 100, k)
        (labels, centers) = CC.kmeans_LSH(100, matrix, bands, max_iters=300)
        print average_sim(labels, matrix)

    def test_standard():
        db = cluster.KMeans(n_clusters=100, max_iter=300)
        labels = db.fit_predict(matrix)
        print average_sim(labels, matrix)

    #cProfile.run("test_standard()")
    cProfile.run("test_LSH(bands)")
    #cProfile.run("test_JLT(100)")
    #cProfile.run("test_SVD(60)")
    #cProfile.run("test_JLT()")
    #cProfile.run("test_SVD()")

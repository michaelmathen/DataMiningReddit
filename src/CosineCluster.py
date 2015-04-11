import json
from scipy.sparse import csr_matrix, diags, coo_matrix, vstack
from scipy.sparse.linalg import svds
import sklearn.cluster as cluster
import os
import numpy as np
import numpy.linalg as npl
from numpy.random import normal
import heapq
import math
from random import randint, random
from scipy.linalg import hadamard
import cProfile

def matrix_repre(directory):
    """
    Constructs a matrix corresponding to the subreddits that users have posted to.
    """
    file_names = os.listdir(directory)
    subreddit_map = {}
    reverse_map = []
    all_rows = []
    all_cols = []
    data = []
    row_i = 0
    for fname in file_names:
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                counts = json.load(f)
                user_name = fname[:-len(".json")]
                for subreddit in counts:
                    if subreddit not in subreddit_map:
                        subreddit_map[subreddit] = len(subreddit_map)
                        reverse_map.append(subreddit)
                    all_rows.append(row_i)
                    all_cols.append(subreddit_map[subreddit])
                    data.append(counts[subreddit])
                row_i += 1
    user_count_matrix = csr_matrix((data, (all_rows, all_cols)))
    return (user_count_matrix, subreddit_map, reverse_map)


def k_means_pp(points, k, initial = None):
    (n, d) = points.shape
    if initial is None:
        offset = 0
        centers = []
        centers.append(points[randint(0, n - 1), :])
    else:
        centers = initial
        offset = len(initial)
    for i in xrange(1, k - offset):
        #get distance to each cluster so
        # kxd and d x n = kxn
        # want 1 x n of distance sum to point
        pt_dist = np.ones((n, 1)) * len(centers)
        for center in centers:
            #center = center.reshape((np.prod(center.shape), 1))
            pt_dist = pt_dist - points.dot(center.transpose())
        probs = 1 / pt_dist.sum() * pt_dist
        cdf = np.ravel(np.cumsum(probs))
        r = random()
        index = np.searchsorted(cdf, r)
        centers.append(points[index - 1, :])
    for center in centers:
        print
        print center.shape
    return vstack(centers, format="csr")

def normalize_rows(matrix):
    """
    Normalize each row in the matrix so that the are of length 1.
    """
    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1))
    def dv(x):
        if x == 0:
            return 0
        else:
            return 1.0 / x
    f = np.vectorize(dv)
    row_norms = np.ravel(f(row_norms))
    row_norms = diags(row_norms, 0)
    matrix = row_norms.dot(matrix)
    return matrix


def argmax(arr):
    (n, k) = arr.shape
    out = np.zeros(n)
    for i in xrange(n):
        (row_i, col_i) = (arr[i,:] == arr[i,:].max()).nonzero()
        out[i] = col_i[0]
    return out


def lsh_h(f_n, d):
    """
    Returns a matrix of random unit vectors
    """
    mat = normalize_rows(normal(0, 1, (f_n, d)))
    return mat.transpose()

def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def approx_bound(eps, n):
    return 1 / eps ** 2 * math.log(n)


def fjlt(n, d, eps, p):
    k = int(approx_bound(eps, n) + .5)
    #Build the Hadamard
    val_len = nextpow2(d)
    H = (1 / math.sqrt(d)) * hadamard(val_len)[0:d, 0:d]

    #Build the diagonal matrix
    diag_mat = np.zeros(d)
    for i in xrange(d):
        if np.random.random() <= 0.5:
            diag_mat[i] = 1
        else:
            diag_mat[i] = -1
    D = np.diag(diag_mat, 0)
    q = min(math.log(n)**p / (d * eps ** (p - 2)), 1.0)

    p_data = []
    p_rows = []
    p_cols = []
    for i in xrange(k):
        for j in xrange(d):
            if  q > np.random.random():
                p_data.append(np.random.normal(0, 1/q))
                p_rows.append(i)
                p_cols.append(j)
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d))
    S = P.dot(H).dot(D)
    return S


def kmeans_cosine(k, sparse_matrix,  max_iters=100):
    sparse_matrix = normalize_rows(sparse_matrix)
    U, s, V = svds(sparse_matrix, k=25)
    #theta = fjlt(n, d, .01, 2)
    sparse_matrix = sparse_matrix.dot(V.transpose())
    sparse_matrix = csr_matrix(sparse_matrix)
    centers = k_means_pp(sparse_matrix, k)
    distances = sparse_matrix.dot(centers.transpose())
    labels = argmax(distances)
    
    for i in xrange(max_iters):
        unqf = np.unique(labels)
        new_centers= []
        for label in unqf:
            #Find average vector and normalize it onto sphere surface
            center = sparse_matrix[labels == label, :].sum(axis=0)
            new_centers.append(center / npl.norm(center))
        new_centers = k_means_pp(sparse_matrix, k, initial=new_centers)
        print new_centers.shape
        distances = sparse_matrix.dot(new_centers.transpose())
        new_labels = argmax(distances)

        if np.array_equal(new_labels, labels):
            return (labels, new_centers)

        centers = new_centers
        labels = new_labels

    return (labels, centers)

def cosine_cluster_users_matrix(user_count_matrix, k = 60):
    """
    Just clusters the matrix using cosine similiarity.
    If we normalize the rows of the matrix then 
    ||a - b||^2 = (a - b)(a - b)^T = ||a||^2 + ||b||^2 - 2a^Tb = 2(1 - a^Tb)
    = 2(1 - cos(\theta))
    """
    user_count_matrix = normalize_rows(user_count_matrix)
    #db = cluster.DBSCAN(min_samples=5, algorithm='brute', metric="cosine")
    #db = cluster.AgglomerativeClustering(n_clusters=30)#, affinity="cosine", linkage="complete")
    #db = cluster.MiniBatchKMeans(n_clusters=k, max_iter=1000, batch_size=400)
    db = cluster.KMeans(n_clusters=k, max_iter=10000, init='random')
    #db = cluster.SpectralClustering(n_clusters=30)
    #db = cluster.Ward(n_clusters=30)
    #db = cluster.Birch(n_clusters=30)
    return db.fit_predict(user_count_matrix)

def cluster_center(matrix, labels, label):
    cluster_rows = matrix[labels == label, :]
    return cluster_rows.sum(axis=0) / float(cluster_rows.sum())

def cluster_centers(count_matrix, reverse_map, labels):
        
    #Now we have to make sense of what each cluster actually is.
    cluster_bags = {}
    for label in np.unique(labels):
        center = cluster_center(count_matrix, labels, label)
        center = np.ravel(center)
        ixs = np.nonzero(center > 0)[0]
        cluster_bags[label] = {reverse_map[ix]:center[ix] for ix in ixs}
    return cluster_bags

def threshold_bag(bag, epsilon):
    total = sum([bag[sub] for sub in bag])
    thresh_bag = {}
    for sub in bag:
        if bag[sub] > total * epsilon:
            thresh_bag[sub] = bag[sub]
    return thresh_bag
        

def k_representatives(bag, k):
    if bag > k:
        values = heapq.nlargest(k, bag, key=bag.__getitem__)
        return [(value, bag[value]) for value in values]
    else:
        return bags[bag_key].items()
    
(matrix, subreddit_map, reverse_map) = matrix_repre("../data/subreddits")
#matrix = matrix > 0
arr = np.zeros(len(subreddit_map))
arr[subreddit_map["darksouls"]] = 1
matrix = matrix[matrix.dot(arr) > 0, :]

#labels = cosine_cluster_users_matrix(matrix, k=150)
#(labels, centers) = kmeans_cosine(10, matrix)
def run():
    (labels, centers) = kmeans_cosine(10, matrix, max_iters=1)
#cProfile.run("run()")

(labels, centers) = kmeans_cosine(4, matrix, max_iters=100)

bags = cluster_centers(matrix, reverse_map, labels)
print (matrix != 0).sum() / float(np.product(matrix.shape))

print matrix.shape

key_order = sorted(bags, key=lambda bag_key: sum(labels == bag_key))
for bag_key in key_order:
    strs = map(lambda x: str(x[0]) + ": %.3f"%x[1], k_representatives(bags[bag_key], 5))
    print str(sum(labels == bag_key)) + " " + " ".join(strs)


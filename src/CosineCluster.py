import json
from scipy.sparse import csr_matrix, diags, vstack
import sklearn.cluster as cluster
import os
import numpy as np
import numpy.linalg as npl
from numpy.random import normal
import heapq
from random import randint, random
import cProfile

from collections import defaultdict

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
        centers = map(csr_matrix, initial)
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
    return vstack(centers, format="csr")


def normalize_rows(matrix):
    """
    Normalize each row in the matrix so that the are of length 1.
    """
    row_norms = np.sqrt(matrix.multiply(matrix)).sum(axis=1)
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


def cosine_hash(seeds, pts):
    """
    Use LSH with cosine similarity to hash 
    rows. Each row is mapped to a set of 
    r integers. 
    Each integer corresponds to the groups that
    the row belongs too.
    A row can belong to multiple groups
    """
    (n, d) = pts.shape
    indices = []
    for band in seeds:
        (d, b_i) = band.shape
        base_mat = (2 ** np.array(range(b_i), ndmin=2)).transpose()
        indices.append((pts.dot(band) > 0).dot(base_mat))
    return indices


def assign_labels(pt_map, center_map):
    k = center_map[0].shape[0]
    n = pt_map[0].shape[0]
    labels = np.ones(pt_map[0].shape[0], dtype=int) * -1
    unq_labels = set()
    #Should take O(kr)
    table = []
    for i in xrange(len(center_map)):
        table.append({})
        for j in xrange(k):
            if center_map[i][j, 0] in table[-1]:
                table[-1][center_map[i][j, 0]].append(j)
            else:
                table[-1][center_map[i][j, 0]] = [j]
    for j in xrange(n):
        label = defaultdict(int)
        for i in xrange(len(center_map)):
            #We have already found this
            key = pt_map[i][j, 0]
            if key in table[i]:
                for el in table[i][key]:
                    label[el] += 1

        if len(label) != 0:
            max_label = max(label, key=label.__getitem__)
            unq_labels.add(max_label)
            labels[j] = max_label
    return (labels, unq_labels)


def kmeans_LSH(k, sparse_matrix, bands, max_iters=100):
    sparse_matrix = normalize_rows(sparse_matrix)
    (n, d) = sparse_matrix.shape
    seeds = [normal(size=(d, b)) for b in bands]
    index_map = cosine_hash(seeds, sparse_matrix)
    centers = k_means_pp(sparse_matrix, k)
    center_map = cosine_hash(seeds, centers)
    (labels, unqf) = assign_labels(index_map, center_map)
    for i in xrange(max_iters):
        new_centers= []
        max_label = -1
        max_label_num = 0
        for label in unqf:
            #Find average vector and normalize it onto sphere surface
            labeled_rows = sparse_matrix[labels == label, :]
            if max_label_num < labeled_rows.shape[0]:
                max_label = label
                max_label_num = labeled_rows.shape[0]
            center = csr_matrix(labeled_rows.sum(axis=0))
            #ncenter = csr_matrix(center / npl.norm(center, norm=2))
            new_centers.append(center)
        #Pick some random points to serve as new cluster centers
        for j in xrange(k - len(new_centers)):
            new_centers.append(csr_matrix(sparse_matrix[randint(0, n - 1), :]))
        new_centers = vstack(new_centers, format="csr")
        center_map = cosine_hash(seeds, new_centers)
        (new_labels, unqf) = assign_labels(index_map, center_map)
        print i
        if np.array_equal(new_labels, labels):
            return (labels, new_centers)

        centers = new_centers
        labels = new_labels

    return (labels, centers)


def kmeans_cosine(k, sparse_matrix,  max_iters=100):
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
        distances = sparse_matrix.dot(new_centers.transpose())
        new_labels = argmax(distances)

        if np.array_equal(new_labels, labels):
            return (labels, new_centers)

        centers = new_centers
        labels = new_labels

    return (labels, centers)


def cluster_users_matrix(user_count_matrix, k = 60):
    """
    Just clusters the matrix using cosine similiarity.
    If we normalize the rows of the matrix then 
    ||a - b||^2 = (a - b)(a - b)^T = ||a||^2 + ||b||^2 - 2a^Tb = 2(1 - a^Tb)
    = 2(1 - cos(\theta))
    """
    matrix = normalize_rows(user_count_matrix)
    #db = cluster.KMeans(n_clusters=k, max_iter=10000)
    db = cluster.MiniBatchKMeans(n_clusters=k, batch_size=10000, max_iter=10000)
    return db.fit_predict(matrix)

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

def average_sim(features):
    (n, d) = features.shape
    center = features.sum(axis=0)
    ncenter = csr_matrix(center / npl.norm(center, ord=2))
    return features.dot(ncenter.transpose()).sum() / float(n)


def cluster_sim(features):
    (n, d) = features.shape
    center = features.sum(axis=0)
    ncenter = csr_matrix(center / npl.norm(center, ord=2))
    return features.dot(ncenter.transpose()).sum()


if __name__ == "__main__":
    (matrix, subreddit_map, reverse_map) = matrix_repre("../data/subreddits")
    matrix = matrix[np.ravel(matrix.sum(axis=1) > 0), :]
    print matrix.shape
    matrix = normalize_rows(matrix)
    print average_sim(matrix)
    """
    mask = np.zeros(matrix.shape[0], dtype=bool)
    mask[subreddit_map["AskReddit"]] = True
    rows = matrix[:, mask] > 0
    rows = np.ravel(rows.todense())
    matrix = matrix[rows, :]
    """
    #matrix = csr_matrix(matrix, dtype=float)
    labels = cluster_users_matrix(matrix, k = 100)

    #bands = [5*i - 4 for i in xrange(1, 13)][::-1]
    bands = [12] * 75  
    #print bands
    print (matrix != 0).sum() / float(np.product(matrix.shape))
    #(labels, centers) = kmeans_LSH(120, matrix, bands, max_iters=300)
    bags = cluster_centers(matrix, reverse_map, labels)

    key_order = sorted(bags, key=lambda bag_key: sum(labels == bag_key))
    for bag_key in key_order:
        print average_sim(matrix[labels == bag_key,:])
        strs = map(lambda x: str(x[0]) + ": %.3f" % x[1], k_representatives(bags[bag_key], 10))
        print str(sum(labels == bag_key)) + " " + " ".join(strs)



#cProfile.run("run()")


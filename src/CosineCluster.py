import json
from scipy.sparse import csr_matrix, diags
import sklearn.cluster as cluster
import os
import numpy as np
import heapq


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
    #db = cluster.MiniBatchKMeans(n_clusters=30, max_iter=1000, batch_size=400)
    db = cluster.KMeans(n_clusters=k, max_iter=10000)
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
arr[subreddit_map["programming"]] = 1
matrix = matrix[matrix.dot(arr) > 0, :]

labels = cosine_cluster_users_matrix(matrix, k=10)
bags = cluster_centers(matrix, reverse_map, labels)

print (matrix != 0).sum() / float(np.product(matrix.shape))
print matrix.shape

key_order = sorted(bags, key=lambda bag_key: sum(labels == bag_key))
for bag_key in key_order:
    strs = map(lambda x: str(x[0]) + ": %.3f"%x[1], k_representatives(bags[bag_key], 5))
    print str(sum(labels == bag_key)) + " " + " ".join(strs)
"""
#Just linguists
labels = cosine_cluster_users_matrix(lingmat, k=3)
bags = most_important(lingmat, reverse_map, labels)

key_order = sorted(bags, key=lambda bag_key: sum(labels == bag_key))
for bag_key in key_order:
    print str(sum(labels == bag_key)) + " " + str(k_representatives(bags[bag_key], 10))

"""

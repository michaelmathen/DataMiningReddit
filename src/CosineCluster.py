import json
from scipy.sparse import csr_matrix, diags
import sklearn.cluster as cluster
import os
import numpy as np
import heapq


def cluster_directory(directory):
    """
    Does a density based cluster on the user data using cosine similiarity.
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
    row_norms = np.sqrt(user_count_matrix.multiply(user_count_matrix).sum(axis=1))
    def dv(x):
        if x == 0:
            return 0
        else:
            return 1.0 / x
    f = np.vectorize(dv)
    row_norms = np.ravel(f(row_norms))
    print row_norms
    row_norms = diags(row_norms, 0)
    user_count_matrix = row_norms.dot(user_count_matrix)
    
    #user_count_matrix
    #user_count_matrix = user_count_matrix.log1p()
    print user_count_matrix.shape
    #db = cluster.DBSCAN(min_samples=5, algorithm='brute', metric="cosine")
    db = cluster.AgglomerativeClustering(n_clusters=30)#, affinity="cosine", linkage="complete")
    #db = cluster.KMeans(n_clusters=23)
    labels = db.fit_predict(user_count_matrix.todense())
    #Now we have to make sense of what each cluster actually is.
    cluster_bags = {}
    for row in xrange(user_count_matrix.shape[0]):
        (rows, cols) = user_count_matrix.getrow(row).nonzero()
        for col in cols:
            subreddit = reverse_map[col]
            count = user_count_matrix[row, col]
            if labels[row] not in cluster_bags:
                cluster_bags[labels[row]] = {}
            if subreddit not in cluster_bags[labels[row]]:
                cluster_bags[labels[row]][subreddit] = count
            else:
                cluster_bags[labels[row]][subreddit] += count
    return (cluster_bags, labels, user_count_matrix)

def threshold_bag(bag, epsilon):
    total = sum([bag[sub] for sub in bag])
    thresh_bag = {}
    for sub in bag:
        if bag[sub] > total * epsilon:
            thresh_bag[sub] = bag[sub]
    return thresh_bag
        

def k_representatives(bags, k):
    representatives = []
    for bag_key in bags:
        if bags[bag_key] > k:
            representatives.append(heapq.nlargest(k, bags[bag_key], key=bags[bag_key].__getitem__))
        else:
            representatives.append(bags[bag_key].keys())
    return representatives
    

(bags, labels, matrix) = cluster_directory("../data/subreddits/")

print (matrix != 0).sum() / float(np.product(matrix.shape))

for repre, bag_key in zip(k_representatives(bags, 5), bags):
    print str(repre) + " " + str(sum(labels == bag_key))

#data = cluster_directory("../data/subreddits/")[0]

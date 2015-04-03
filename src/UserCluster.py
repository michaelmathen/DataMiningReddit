import sklearn.cluster as cluster
from RedditParser import All_User_Comments, comment_map, Fast_User_Comments
#import frequent_items as fq
from  scipy.sparse import csr_matrix
import requests

def MG(stream, k):
    counters = {}
    for item in stream:
        if item in counters:
            counters[item] += 1
        elif len(counters) < (k - 1):
            counters[item] = 1
        else:
            for key in counters.keys():
                counters[key] -= 1
                if counters[key] <= 0:
                    counters.pop(key)
    return counters


def cluster_by_subreddit(user_name_streams, epsilon):
    subred = comment_map["subreddit"]
    indices = []
    data = []
    all_subreddits = {}
    row_i = 0
    for user_name in user_name_streams:
        print user_name
        try:
            user_counts = MG(Fast_User_Comments(user_name, max_pages=4), int(1/epsilon) + 1)
            for subreddit in user_counts:
                if subreddit not in all_subreddits:
                    all_subreddits[subreddit] = len(all_subreddits)
                indices.append((row_i,
                                all_subreddits[subreddit]))
                data.append(user_counts[subreddit])
            row_i += 1
            print user_counts
        except requests.exceptions.HTTPError as e:
            continue
    all_user_counts = csr_matrix((data, indices))
    labels = cluster.DBSCAN(min_samples=5, metric="cosine").fit_predict(all_user_counts)
    return labels

if __name__ == "__main__":
    g = (name.rstrip() for name in open("../data/user_names_all_1000.csv"))
    print cluster_by_subreddit(g, .01)
    #for el in Fast_User_Comments("sunbolts"):


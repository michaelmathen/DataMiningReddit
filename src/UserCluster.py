#import sklearn.cluster as cluster
from RedditParser import All_User_Comments, comment_map, Fast_User_Comments
#import frequent_items as fq
#from  scipy.sparse import csr_matrix
import requests
import json
import os

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


def get_subreddits(user_name_streams, epsilon):
    for user_name in user_name_streams:
        try:
            user_counts = MG(Fast_User_Comments(user_name, max_pages=16), int(1/epsilon) + 1)
            yield (user_name, user_counts)
        except requests.exceptions.HTTPError as e:
            continue

def sanitize(directory, file_name):
    file_names = os.listdir(directory)
    names = set()
    for fname in file_names:
        if fname.endswith(".json"):
            names.add(fname[:-len(".json")])
    with open(file_name) as f:
        for name in f:
            name = name.rstrip()
            if  name not in names:
                yield name
    

if __name__ == "__main__":
    for (user, counts) in get_subreddits(sanitize("../data/subreddits/", "../data/user_names_rand.txt"), .001):
        with open("../data/subreddits/" + user + ".json", "w+") as f:
            print user
            f.write(json.dumps(counts) + "\n")

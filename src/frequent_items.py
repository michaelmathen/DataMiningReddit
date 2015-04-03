import numpy as np
from random import random

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

def exact(stream):
    counters = {}
    for item in stream:
        if item in counters:
            counters[item] += 1
        else:
            counters[item] = 1
    return counters

def make_multiplicative_hash(K):
    a = random()
    def random_hash_func(str_b):
        k = hash(str_b)
        return int(K * (a * k - int(a * k)))
    return random_hash_func


def stream_length(stream):
    n = 0
    for el in stream:
        n += 1
    return n

h = [make_multiplicative_hash(10) for i in xrange(5)]

def Count_Min(stream, t, k):
    C = np.zeros((t, k))
    for item in stream:
        for j in xrange(t):
            C[j, h[j](item)] += 1
    return (C, h)

def query_CM_sketch(item, C, h):
    row_j = min(xrange(len(h)), key=lambda j: C[j, h[j](item)])
    return C[row_j, h[row_j](item)]

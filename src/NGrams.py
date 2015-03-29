
from itertools import *
from sets import Set
from collections import deque
from random import random
from sys import maxint
from time import time
import math
import RedditParser
import re


def parse_body(str):
    str.encode('ascii',errors = 'ignore')
    str = re.sub('[?!@#$,/.<>()*&^%#@~+=]', '', str)
    ans = str.split(" ", str.count(" "))
    return ans




if __name__== "__main__":
    str = "hello, poop, is! so? Great<> yeah."
    print(str)
    fin = parse_body(str)





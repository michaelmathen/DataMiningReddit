
from itertools import *
from sets import Set
from collections import deque
from random import random
from sys import maxint
from time import time
import math
import RedditParser
import re
import nltk

"""
*This method takes in a string, takes only the ascii values, and removes weird characters
*It then returns a list of all of the strings broken up by spaces
"""
def parse_body(str):
    str.encode('ascii',errors = 'ignore')#converts the unicode to ascii ignoring some characters
    str = re.sub('[?!@#$,/.<>()*&^%#@~+=]', '', str)#removes special characters
    ans = str.split(" ", str.count(" "))#splits the string besed on words
    return ans

"""
*This method returns numPost posts that have a score above thresh
*lowerR is an array of the lower bounds for the set 
"""
def top_n_posts_range(numPosts,r1):
    #create a generator for the top reddit posts
    topCom = RedditParser.Parse_Subreddit("all",order = RedditParser.Get_Top_From_All)#use .next() to get other elem
    iter = 0  #check on the outer while loop
    count = 0 #number of posts that we have accepted
    r1Iter = 0 #move through the r1 array
    posts = []
    while(count < numPosts):
        temp = topCom.next()
        if(temp[2] > r1[0] and temp[2] <  r1[1]):
            posts.append(temp)
            count += 1
            iter += 1

    return posts

def top_n_posts(numPosts,r1):
    #create a generator for the top reddit posts
    topCom = RedditParser.Parse_Subreddit("all",order = RedditParser.Get_Top_All)#use .next() to get other elem
    #iter = 0  #check on the outer while loop
    count = 0 #number of posts that we have accepted
   # r1Iter = 0 #move through the r1 array
    posts = []
    while(count < numPosts):
        temp = topCom.next()
        if(temp[2] > r1):
            posts.append(temp)
            count += 1
           # iter += 1

    return posts

def top_n_posts_spec(numPosts,r1,spec):
    #create a generator for the top reddit posts
    topCom = RedditParser.Parse_Subreddit(spec,order = RedditParser.Get_Top_All)#use .next() to get other elem
    #iter = 0  #check on the outer while loop
    count = 0 #number of posts that we have accepted
   # r1Iter = 0 #move through the r1 array
    posts = []

    for temp in topCom:
        if(temp[2] > r1):
            posts.append(temp)
               

    return posts

def calc_ngrams(arrSent,gramSize):
    gramArr = []
    for i in range(len(arrSent)):
        gram = Set(nltk.ngrams(parse_body(arrSent[i][0]),gramSize))
        gramArr.append(gram)
        
    return gramArr

def jacc_sim(s1,s2):
    return len(s1 & s2) / (float(len(s1 | s2))+1)

def jacc_compare(grams):
    jsim = 0.0
    for i  in range(len(grams)-1):
        for j in range(i,len(grams)):
            jsim += jacc_sim(grams[i],grams[j])
    return jsim / nCr(len(grams),2)

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def avg_post(numPosts):
    avg = 0
    topcom = RedditParser.Parse_Subreddit("all",order = RedditParser.Get_Top_All)
    for i in range(numPosts):
        temp = topcom.next()
        avg += temp[2]
        #print(temp[2])
    return (avg / numPosts)

def test(numPosts,thresh,ty):
    posts = top_n_posts_spec(numPosts,thresh,ty)
    print(ty)
    print(len(posts))
    grams = calc_ngrams(posts,1)
    print("1-grams")
    print(jacc_compare(grams))

    grams = calc_ngrams(posts,2)
    print("2-grams")
    print(jacc_compare(grams))
    
    grams = calc_ngrams(posts,3)
    print("3-grams")
    print(jacc_compare(grams))

    grams = calc_ngrams(posts,4)
    print("4-grams")
    print(jacc_compare(grams))
            

def main():
    str = "hello, poop!"
    print(str)
    fin = parse_body(str)
   # tempPost = RedditParser.All_User_Comments("_vargas_").next()
   # print(parse_body(tempPost[0]))
   # print(parse_body(tempPost[0])[0])
   # print(tempPost[2])
   # print(avg_post(1000))

   # print(jacc_sim(grams[0],grams[0]))
   # print(jacc_sim(grams[0],grams[1]))
    """
    print("All above 1000")
    posts = top_n_posts(100,1000)
    print(posts[0][0])
    print(posts[9][0])
    print(parse_body(posts[0][0]))
    print(parse_body(posts[9][0]))

    #Run N-grams on normal sets
    
    grams = calc_ngrams(posts,1)
    print("grams")
    print(jacc_compare(grams))

    grams = calc_ngrams(posts,2)
    print("grams")
    print(jacc_compare(grams))
    
    grams = calc_ngrams(posts,3)
    print("grams")
    print(jacc_compare(grams))

    grams = calc_ngrams(posts,4)
    print("grams")
    print(jacc_compare(grams))
   
    print("100 - 200")
    posts = top_n_posts_range(1000,[100,200])
    grams = calc_ngrams(posts,1)
    print("1-grams")
    print(jacc_compare(grams))

    grams = calc_ngrams(posts,2)
    print("2-grams")
    print(jacc_compare(grams))
    
    grams = calc_ngrams(posts,3)
    print("3-grams")
    print(jacc_compare(grams))

    grams = calc_ngrams(posts,4)
    print("4-grams")
    print(jacc_compare(grams))

    """

    #Run N-Grams on specific subreddits for similarity
    """
    Here we See if there are more similarities between specific subreddits for example funny 
    """

    test(300,100,'funny')
    test(300,100,'worldnews')
    test(300,100,'science')

    print('over 1000')

    test(300,1000,'funny')
    test(300,1000,'worldnews')
    test(300,1000,'science')

    #serious, politics, science
    
    
   
    


if __name__== "__main__":
    main()

    





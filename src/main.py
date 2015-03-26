import RedditParser
            

if __name__ == "__main__":
    for comment in RedditParser.Parse_Subreddit("math"):
        print comment


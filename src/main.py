import RedditParser
            

if __name__ == "__main__":
    #r = RedditParser.praw.Reddit(RedditParser.USER_STR)
    #print r.get_submission(submission_id='20gewk')
    for comment in RedditParser.All_User_Comments("_vargas_"):
        print comment

        
        #for comment in RedditParser.Parse_Subreddit("math"):
        #print comment


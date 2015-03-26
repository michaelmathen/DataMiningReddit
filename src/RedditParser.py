import praw
from itertools import count

USER_STR = 'OSX:Data Mining School Project Bot:v1.0 by /u/haskellmonk'
comment_map = dict(zip(["body",
                        "gilded",
                        "score",
                        "utc",
                        "subreddit",
                        "author",
                        "submission_title",
                        "submission_author",
                        "submission_score",
                        "submission_utc"],
                       count()))


def Parse_Subreddit(name):
    r = praw.Reddit(USER_STR)
    for submission in r.get_subreddit(name).get_hot(limit=None):
        for comment_obj in praw.helpers.flatten_tree(submission.comments):
            if isinstance(comment_obj, praw.objects.MoreComments):
                #I am just ignoring these
                continue
            try:
                comment_data = (comment_obj.body, 
                                comment_obj.gilded,
                                comment_obj.score,
                                comment_obj.created_utc,
                                comment_obj.subreddit.display_name,
                                comment_obj.author.name,
                                submission.title,
                                submission.author.name,
                                submission.score,
                                submission.created_utc)
                yield comment_data
            except:
                print "Bad comment. I am not sure why this happens?"
                continue

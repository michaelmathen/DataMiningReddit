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


def Get_Hot(subreddit_handle):
    return subreddit_handle.get_hot(limit=None)


def Get_New(subreddit_handle):
    return subreddit_handle.get_new(limit=None)


def Get_Top(subreddit_handle):
    return subreddit_handle.get_top(limit=None)


def Get_Rising(subreddit_handle):
    return subreddit_handle.get_rising(limit=None)


def Get_Controversial(subreddit_handle):
    return subreddit_handle.get_controversial(limit=None)




def Parse_Subreddit(name, order=Get_Hot):
    """
    Given the name of a subreddit get all comments ever posted in that subreddit.
    The optional order determines the order in which they are returned.
    The functions above can be passed in to change the order.
    """
    r = praw.Reddit(USER_STR)
    for submission in order(r.get_subreddit(name)):
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


def All_User_Comments(user_name, sort='new'):
    """
    Give an username this function will return all comments that the user 
    has ever posted formatted in the same way that we use for Parse_Subreddit.
    user_name -- The name of the user.
    sort -- The order in which the comments are returned.
    """
    r = praw.Reddit(USER_STR)
    for comment_obj in r.get_redditor(user_name).get_comments(sort=sort,limit=None):
        try:
            if isinstance(comment_obj, praw.objects.MoreComments):
                continue
            submission= r.get_submission(submission_id=comment_obj.link_id[3:])
            comment_data = (comment_obj.body, 
                            comment_obj.gilded,
                            comment_obj.score,
                            comment_obj.created_utc,
                            comment_obj.subreddit.display_name,
                            comment_obj.author.name,
                            comment_obj.link_title,
                            comment_obj.link_author,
                            submission.score,
                            submission.created_utc)
            yield comment_data
        except:
            print "Bad comment."
            continue
        

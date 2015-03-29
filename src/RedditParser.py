import praw
from itertools import count

USER_STR = 'OSX:Data Mining School Project :v1.0 by /u/haskellmonk'
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


def Parse_Subreddit(name, order=Get_Top):
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
            if comment_obj.author is None:
                continue
            if submission.author is None:
                continue
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


def All_Users(subreddit, max_user_number=100):
    """
    Get all the usernames in a subreddit as a stream.
    """
    users = {}
    try:
        for elem in Parse_Subreddit(subreddit, order=Get_Top):
            if elem[comment_map["author"]] in users:
                users[elem[comment_map["author"]]] += 1
            else:
                users[elem[comment_map["author"]]] = 1
            if len(users) >= max_user_number:
                break
            if len(users) % 100 == 0:
                print len(users)
    except Exception:
        return users
    return users


def Stream_To_File(stream, fname, k=10000):
    """
    Keep streaming data to a file from reddit until k is reached or until
    the stream is exhausted.
    """
    with open(fname, "w+") as f:
        f.write("DataStream = [")
        while True:
            try:
                element = stream.next()
                f.write(repr(element))
                f.write(",\n")
            except:
                f.write("]\n")
                break



def All_User_Comments(user_name, sort='new'):
    """
    Give an username this function will return all comments that the user 
    has ever posted formatted in the same way that we use for Parse_Subreddit.
    user_name -- The name of the user.
    sort -- The order in which the comments are returned.
    """
    r = praw.Reddit(USER_STR)
    for comment_obj in r.get_redditor(user_name).get_comments(sort=sort,limit=None):
        if isinstance(comment_obj, praw.objects.MoreComments):
            continue
        if comment_obj.author is None:
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

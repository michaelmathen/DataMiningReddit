import csv
import RedditParser


def Save_Reddit_User_Names(subreddit, number):
    user_counts = RedditParser.All_Users(subreddit, max_user_number=number)
    with open("user_names_%s_%d.csv" % (subreddit, len(user_counts)), "w+") as f:
        print "Finished getting user counts"
        writer = csv.writer(f, delimiter=',', quotechar='\'', quoting=csv.QUOTE_ALL)
        for user in user_counts:
            writer.writerow([user, user_counts[user]])
    print "Finished writing names"



if __name__ == "__main__":
    Save_Reddit_User_Names("all", 50000)
    




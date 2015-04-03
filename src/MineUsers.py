import RedditParser


def Save_Reddit_User_Names(file_name, subreddit, number):
    name_set = set()
    try:
        with open(file_name, "r") as f:
            for line in f:
                name_set.add(line.rstrip())
    except:
        pass
    with open(file_name, "a+") as f:
        k = len(name_set)
        for user in RedditParser.All_Users(subreddit, max_user_number=number,
                                           users=name_set):
            f.write(user + "\n")
            k += 1
            if k % 100 == 0:
                print k
            if k == number:
                break    



if __name__ == "__main__":
    Save_Reddit_User_Names("user_names.txt", "fffffffuuuuuuuuuuuu", 400000)

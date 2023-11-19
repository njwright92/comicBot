import praw

# Initialize PRAW with your credentials
reddit = praw.Reddit(
    client_id='6xZmR28aYQv92M4p5EgXlw',  # Replace with your client_id
    client_secret='4Ym_KbdxlTbMgTD5Nqo1MaEjyy98Mw',  # Replace with your client_secret
    user_agent='comedify'  # Your app's name
)

subreddits = ["dadjokes", "Jokes"]

# Open a file to write jokes
with open("redditjokes.txt", "w", encoding='utf-8') as file:
    for subreddit in subreddits:
        file.write(f"--- Jokes from r/{subreddit} ---\n")
        # Fetching 100 posts
        for submission in reddit.subreddit(subreddit).hot(limit=100):
            file.write(submission.title + "\n")
            file.write(submission.selftext + "\n\n")

print("Jokes have been written to redditjokes.txt")

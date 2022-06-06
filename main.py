from numpy import sort
import praw
import os
import markovify
from psaw import PushshiftAPI
from dotenv import load_dotenv
from datetime import date
from functools import cmp_to_key

load_dotenv()

subr = 'subredditbots'

#comparator
def compare(x, y):
    return x.upvote_ratio - y.upvote_ratio

#praw initialization
reddit = praw.Reddit(
    #client_id=os.getenv('REDDIT_CLIENT_ID'),
    #client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    client_id="VefkO3XJsMBeKX6xycc1bA",
    client_secret="rvjCJGN4jmf-B1WEQp35YXwofGWBMw",
    user_agent="Bot for simulating /r/askreddit content",
    username="ask_reddit_bot_0x001",
    password="LvH65Lq%pzu6F79",
    #password=os.getenv('REDDIT_PASSWORD'),
    #refresh_token=os.getenv('REFRESH_TOKEN'),
    #redirect_uri="http://localhost:8080"
)

#print(reddit.user.me())

#pushshift query
api = PushshiftAPI()
results = list(api.search_submissions(
    limit=1000,
    mod_removed = False,
    subreddit = "askreddit"
))
top_results = results
sorted(top_results, key=cmp_to_key(compare))
top_results=top_results[-20:]

#write data to file
data = open("data.txt", "w")
for result in results:
    data.write(result.title)
    data.write("\n")
data.close()

data = open("top_data.txt", "w")
for result in top_results:
    data.write(result.title)
    data.write("\n")
data.close()

#markovify api
with open("data.txt") as f:
    text = f.read()
text_model_reg = markovify.Text(text)

with open("top_data.txt") as f:
    text = f.read()
text_model_top = markovify.Text(text)

text_model = markovify.combine([text_model_reg, text_model_top], [1, 2])

#write to output file
data = open("generated_content.txt", "w")
data.write(result.title)
data.write("\n")

tmp = str(text_model.make_sentence(tries=100, max_overlap_ratio=0.5))
while tmp == None:
    tmp = str(text_model.make_sentence(tries=100, test_output = False))

#post to reddit using praw api
selftext=""
#reddit.subreddit(subr).submit(tmp, selftext=selftext)

data.close()

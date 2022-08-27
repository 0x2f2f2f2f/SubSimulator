import tensorflow as tf
import numpy as np
import os
import time
import praw
import json
import markovify
import selftext
import title
from numpy import sort
from psaw import PushshiftAPI
from functools import cmp_to_key

subr = "askreddit"
credentials = 'bots.json'

with open(credentials) as f:
    creds = json.load(f)

#comparator
def compare(x, y):
    return x.upvote_ratio - y.upvote_ratio

#praw initialization
reddit = praw.Reddit(
    client_id=creds['bots']['client_id'],
    client_secret=creds['bots']['client_secret'],
    user_agent="script by u/ask_reddit_bot_0x001",
    username=creds['bots']['username'],
    password=creds['bots']['password'],
    redirect_uri=creds['bots']['redirect_uri'],
)

#pushshift query
api = PushshiftAPI()
initial_query = list(api.search_submissions(
    limit=100,
    mod_removed = False,
    subreddit = subr
))
titles = []
selftexts = []
for query in initial_query:
    titles.append(query)
    if hasattr(query, 'selftext'):
        selftexts.append(query.selftext)
title_ret = title.gen_title(titles)
selftext_ret = selftext.gen_selftext(selftexts)

#post to reddit using praw api
#reddit.subreddit(subr).submit(title_ret, selftext=selftext_ret)


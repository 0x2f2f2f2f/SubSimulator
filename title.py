from ast import List, Str
import tensorflow as tf
import numpy as np
import time
import praw
import json
import markovify
from numpy import sort
from psaw import PushshiftAPI
from functools import cmp_to_key

def gen_title(query_res):
    top_results = []
    for query in query_res:
        if query.upvote_ratio != 1.0:
            top_results.append(query)
    #write data to file
    data = open("data.txt", "w")
    for result in query_res:
        data.write(result)
        data.write("\n")
    data.close()

    data = open("top_data.txt", "w")
    for result in top_results:
        data.write(top_results)
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
    data.write(result)
    data.write("\n")
    data.close()

    title = str(text_model.make_sentence(tries=100, max_overlap_ratio=0.5))
    while title == None:
        title = str(text_model.make_sentence(tries=100, test_output = False))
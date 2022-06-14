import tensorflow as tf
import numpy as np
import time
import praw
import json
import markovify
from numpy import sort
from psaw import PushshiftAPI
from functools import cmp_to_key

def gen_selftext(query_res):
    #open data and obtain vocab for dataset
    text =  open('data.txt', 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    chars = tf.strings.unicode_split(query_res, input_encoding='UTF-8')

    #creating mapping of ids and chars
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    chars = chars_from_ids(ids)
    ids = ids_from_chars(chars)
    tf.strings.reduce_join(chars, axis=-1).numpy()

    #tensor slice fromt he data set
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
import tensorflow as tf
import numpy as np
import time
import praw
import json
import markovify
import mymodel
from numpy import sort
from psaw import PushshiftAPI
from functools import cmp_to_key
from mymodel import MyModel
from onestep import OneStep

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def gen_selftext(query_res):
    #open data and obtain vocab for dataset
    data = open("data.txt", "w")
    for result in query_res:
        data.write(result)
        data.write("\n")
    data.close()
    text =  open('data.txt', 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    chars = tf.strings.unicode_split(query_res, input_encoding='UTF-8')

    #creating mapping of ids and chars
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    ids = ids_from_chars(chars)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    chars = chars_from_ids(ids)
    tf.strings.reduce_join(chars, axis=-1).numpy()

    #tensor slice from the data set
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    #setup tf model
    model = MyModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

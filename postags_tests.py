#!/usr/bin/env python


#%%
import numpy as np
from collections import Counter
from collections import defaultdict
import pandas as pd
import math
import pandas as pd
import string

#%%
def assign_unk(word):
    """Assign tokens to unknown words."""
    punct = set(string.punctuation)

    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee",
                   "ence", "er", "hood", "ion", "ism", "ist", "ity",
                   "ling", "ment", "ness", "or", "ry", "scape",
                   "ship", "ty"]

    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic",
                  "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    if (any(char.isdigit() for char in word)):
        return "--unk_digit--"
    elif any(char in punct for char in word):
        return "--unk_punct--"
    elif any(char.isupper() for char in word):
        return "--unk_upper--"
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_suffix--"
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    else:
        return "--unk--"

#%%
def get_word_tag (line, vocab):
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
    return word, tag

#%%
def preprocess(vocab, data):
    """Preprocess the data"""
    orig = []
    prep = []

    with open(data, 'r') as df:
        for cnt, word in enumerate(df):
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue
            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data, 'r').readlines()))
    assert(len(prep) == len(open(data, 'r').readlines()))
    return orig, prep

#%%
def preprocess2(vocab, data):
    """Preprocess the data"""
    orig = []
    prep = []

    with open(data, 'r') as df:
        for cnt, word in enumerate(df):
            if not word.split():
                word = "--n--"
            elif word.strip() not in vocab:
                word = assign_unk(word)

            orig.append(word.strip())
            prep.append(word.strip())

    assert(len(orig) == len(open(data, 'r').readlines()))
    assert(len(prep) == len(open(data, 'r').readlines()))
    return orig, prep

#%%
def pmatrix(matrix, index, cols=None):
    if cols is None: cols = index
    print(pd.DataFrame(matrix, index=index, columns=cols))

#%% Define some tags
tags = ['RB', 'NN', 'TO' ]

# Where to obtain these transitions?
transition_counts = {
        ('NN', 'NN'): 16241,
        ('RB', 'RB'): 2263,
        ('TO', 'TO'): 2,
        ('NN', 'TO'): 5256,
        ('RB', 'TO'): 855,
        ('TO', 'NN'): 734,
        ('NN', 'RB'): 2431,
        ('RB', 'NN'): 358,
        ('TO', 'RB'): 200,
        }

#%%
num_tags = len(tags)
sorted_tags = sorted(tags)
transition_matrix = np.zeros((num_tags, num_tags))

#%%
for i in range(num_tags):
    for j in range(num_tags):
        tag_tuple = (sorted_tags[i], sorted_tags[j])
        transition_matrix[i, j] = transition_counts.get(tag_tuple, 0)

#%%
rows_sum = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix /= rows_sum

#%%
d = np.diag(transition_matrix).copy()
d = np.expand_dims(d, 1)
d += np.log(rows_sum)

#%%
np.fill_diagonal(transition_matrix, d)

#%%
pmatrix(transition_matrix, sorted_tags, sorted_tags)

#%%
with open('data/WSJ_02-21.pos', 'r') as fd:
    lines = fd.readlines()

#%%
words = [ line.split('\t')[0] for line in lines ]

#%%
freq = Counter(words)

#%%
vocab = [k for k, v in freq.items() if (v > 1 and k != '\n')]
vocab.sort()


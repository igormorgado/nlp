##
import numpy as np
import pandas as pd
import collections
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

##
def text_to_list(corpus):
    return re.findall(r"[\w']+", corpus.lower())

##
def stopwords_remove(corpus_list, stopwords):
    corpus_list_clean = []
    for word in corpus_list:
        if word not in stopwords:
            corpus_list_clean.append(word)
    return corpus_list_clean

##
def ngram_frequency(corpus, rank):
    return Counter(corpus.split(' '))

##
def compute_probability(word_frequency):
    total = sum(word_frequency.values())
    for k, v in word_frequency.items():
        word_frequency[k] = v/total
    return total

## 
def shannon_entropy(values):
    assert abs(sum(values) - 1.0) < 1e-7
    assert min(values) > 0.0
    return sum(values * np.log2(1/values))

##
def ngram(sentences, rank):
    parts = []
    for sentence in sentences:
        words = sentence.split(" ")
        nwords = len(words)
        assert nwords >= rank, f"{rank}-gram can't be created in {nwords} words sentence"
        for i in range(nwords-rank+1):
            parts.append(words[i:i+rank])
    return parts

##
sentences = ["<s> I am Sam </s>",
             "<s> Sam I am </s>",
             "<s> I do not like green eggs and ham </s>"]

## Compute all ngrams to rank 2
max_rank = 2
grams = []
grams_count = []
for rank in range(1,max_rank+1):
    gram_list = ngram(sentences, rank)
    grams.append([ ' '.join(g) for g in gram_list ])
    grams_count.append(Counter(grams[-1]))

## Some Probability checks in Language Model (LM)
ptests = [['I', '<s>'],
          ['Sam', '<s>'],
          ['am', 'I'],
          ['</s>', 'Sam'],
          ['Sam', 'am'],
          ['do', 'I']]

## Compute the probabilities
for p in ptests:
    pstr = f'P ({p[0]} | {p[1]})'
    res = grams_count[1][f'{p[1]} {p[0]}'] / grams_count[0][p[1]]
    print(f'{pstr:14s} = {res:.2f}')


## More realistic sentences
restaurant = ["<s> can you tell me about any good cantonese restaurants close by </s>",
              "<s> mid priced thai food is what i'm looking for </s>",
              "<s> tell me about chez panisse </s>",
              "<s> can you give me a listing of the kinds of food that are available </s>",
              "<s> i'm looking for a good place to eat breakfast </s>", 
              "<s> whan is caffe venezia open during day </s>"]


max_rank = 2
grams = []
grams_count = []
for rank in range(1,max_rank+1):
    gram_list = ngram(restaurant, rank)
    grams.append([ ' '.join(g) for g in gram_list ])
    grams_count.append(Counter(grams[-1]))


# We compute n-gram probabilities, approximating to 2gram probabilities hence
# P(<s> I want english food </s> ==
# P(I | <s> ) * P( want | I) * P (english | want) * P (food | english) * P (</s> | food)
#
# Therefore we can store probabilities in a regular matrix with each element being a prob
# of word in j-column being followed by word in i-row. And fast retrieve this data
#
# Also to avoid underflows and faster computations we will store log p_i in matrix instead
# p_i. Since
#
# log( p1 * p2 * p3 * p3) =  log p1 + log p2 + log p3 + log p4 

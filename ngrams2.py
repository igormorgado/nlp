#!/usr/bin/env python

#%%
import nltk
import re
import random
import numpy as np
import pandas as pd
import math
import string
from collections import defaultdict
from collections import Counter

#%%
nltk.download('punkt')

#%%
def sentence_to_ngram(tokenized_sentence, ngram_order=2):
    assert ngram_order <= len(tokenized_sentence)
    ngram = []
    for i in range(len(tokenized_sentence) - ngram_order + 1):
        ngram.append(tokenized_sentence[i:i+ngram_order])
    return ngram

#%%
def preprocess_ngram_sentence(tokenized_sentence, ngram_order=2):
    return ["<s>"] * (ngram_order-1) + tokenized_sentence + ["</s>"]

#%%
def pretty_print_ngram(tokenized_sentence, ngram_order=2, normalize=True):

    largest_word = max([ len(w) for w in tokenized_sentence])
    digits = len(str(len(tokenized_sentence)))

    sentence = tokenized_sentence.copy()
    if normalize:
        sentence = preprocess_ngram_sentence(sentence, ngram_order)

    ngrams = sentence_to_ngram(sentence, ngram_order)

    for i, ngram in enumerate(ngrams):
        print(f"{i:>{digits}d} ", end='')
        for word in ngram:
            print(f"{word:>{largest_word+1}s} ", end='')
        print()

#%%
def single_pass_trigram_count_matrix(corpus):
    """Creates a trigram count matrix from input corpus

    Input:
        corpus: Pre-processes tokenized corpus

    Return:
        brigrams: list of all bigram prefixes, row index
        vocabulary: list of all found words, the column index
        count_matrix: pandas dataframe with bigram prefixes as rows,
                      vocabulary words as columns
                      and counter of bigram/word combinantion as elements.

    """

    bigrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(dict)

    ngram_order = 3
    for i in range(len(corpus) - ngram_order + 1):
        trigram = tuple(corpus[i:i+ngram_order])

        bigram = trigram[0:-1]
        if not bigram in bigrams:
            bigrams.append(bigram)

        last_word = trigram[-1]
        if not last_word in vocabulary:
            vocabulary.append(last_word)

        key = (bigram, last_word)
        if (bigram, last_word) not in count_matrix_dict:
            count_matrix_dict[key] = 0

        count_matrix_dict[key] += 1

    # Convert count_matrix to np.array to fill blanks
    count_matrix=np.zeros((len(bigrams), len(vocabulary)))

    for trigram_key, trigram_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]), \
                     vocabulary.index(trigram_key[1])] \
                     = trigram_count

    count_matrix = pd.DataFrame(count_matrix, index=bigrams, columns=vocabulary)

    return count_matrix

#%% Smoothing
def add_k_smooting_probability(k, vocabulary_size, n_gram_count, n_gram_prefix_count):
    numerator = n_gram_count + k
    denominator = n_gram_prefix_count + k * vocabulary_size
    return numerator/denominator

#%% Language model evaluation
def train_validation_test_split(data, train_percent, validation_percent, seed=87, shuffle=True):
    """
    SPlits the input data to train/validation/test according to the percentage

    Input:
    data: Preprocesses and tokenized corpus as list of sentences
    train_percent: part to be defined as train in percentage
    validation_percentage: part to be used as percentage.

    Remarks:
        test_set size is 100 - (train_percent + validation_percent)


    Returns:
        train_data
        validation_data
        test_data

    """
    random.seed(seed)

    if shuffle:
        random.shuffle(data)

    train_size = int(len(data) * train_percent / 100)
    train_data = data[0:train_size]

    validation_size = int(len(data) * validation_percent / 100)
    validation_data = data[train_size:train_size + validation_size]

    test_data = data[train_size + validation_size:]

    return train_data, validation_data, test_data

#%% Split sentences
def split_to_sentences(data):
    return data.strip().split('\n')

#%%
def tokenize_sentences(sentences):
    return [ re.findall(r"[\w']+|[.,!?;<=>]", s.lower()) for s in sentences ]

#%% More "professional"
def split_to_sentences_nltk(data):
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sentence_detector.tokenize(data.strip())

def tokenize_sentences_nltk(sentences):
    # Missing better handling of tweeter tags, user marks and symbols
    return [nltk.word_tokenize(s.lower()) for s in sentences]

#%%
def get_tokenized_data(data):
    return tokenize_sentences(split_to_sentences(data))

#%%
def count_words(data):
    return Counter([ word for sentence in data for word in sentence])

#%%
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    word_frequencies = count_words(tokenized_sentences)
    word_frequencies_filtered = dict(filter(lambda x: x[1] >= count_threshold, word_frequencies.items()))
    return list(word_frequencies_filtered.keys())

#%%
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocabulary = set(vocabulary)
    return [[ word if word in vocabulary else unknown_token for word in sentences] for sentences in tokenized_sentences ]

#%%
def preprocess_data(train_data, test_data, count_threshold):
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_pp = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_pp = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_pp, test_data_pp, vocabulary

#%%
# ##########################################
#
# PREDICTOR
#
# ##########################################

#%%
# ##########################################
# Preprocessing
# ##########################################

#%% Read data
with open("data/en_US.twitter.txt", "r") as fd:
    data = fd.read()

#%%
train_data, _, test_data = train_validation_test_split(get_tokenized_data(data), 80, 0, seed=87, shuffle=True)

#%%
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, minimum_freq)


#%%
# ##########################################
# Language Model
# ##########################################

#%%

# #%%
# word_length = [[(w, len(w))  for w in sentence ] for sentence in tokenized]
# word_length_flat = [(w, len(w))  for sentence in tokenized for w in sentence]
#
# #%%
# ngram_order = 5
# ngrams = [sentence_to_ngram(preprocess_ngram_sentence(sentence, ngram_order), ngram_order) for sentence in tokenized]
#
# #%%
# pretty_print_ngram(tokenized[0], 3)
#
# #%%
# count_matrix = single_pass_trigram_count_matrix(tokenized[1])
# bigrams =  count_matrix.index.to_list()
# vocabulary = count_matrix.columns.to_list()
#
# #%% Probability matrix
# row_sums = count_matrix.sum(axis=1)
# prob_matrix = count_matrix.div(row_sums, axis=0)
#
# #%% Check probabilities
# trigram = ('i', 'am', 'happy')
# bigram = trigram[:-1]
# word = trigram[-1]
# trigram_probability = prob_matrix[word][bigram]
#
# #%% lists all words in vocabulary starting with a given prefix
# vocabulary = ['i', 'am', 'happy', 'because', 'learning', '.', 'have', 'you', 'seen','it', '?']
# starts_with = 'ha'
# words_ha = [ w for w in vocabulary if w.startswith(starts_with) ]
#
#
# #%% Perplexity
# # to calculate the exponent, use the following syntax
# # M here is the product of all probabilities of bigrams(Wi|Wi-1)
# p = 10 ** (-250)
# M = 100
# perplexity = p ** (-1/M)
# print(perplexity)
#
#
# #%% Target Vocab size
# M = 3
# word_counts = {'happy': 5,
#                'because': 3,
#                'i': 2,
#                'am': 2,
#                'learning': 3,
#                '.': 1}
#
# #%%
# vocabulary = dict(Counter(word_counts).most_common(M))
# sentence = ['am', 'i', 'learning']
# output_sentence = [ w if w in vocabulary else '<UNK>' for w in sentence ]
#
# #%%
# training_set = ['i', 'am', 'happy', 'because','i', 'am', 'learning', '.']
# training_set_unk = ['i', 'am', '<UNK>', '<UNK>','i', 'am', '<UNK>', '<UNK>']
#
# test_set = ['i', 'am', 'learning']
# test_set_unk = ['i', 'am', '<UNK>']
#
# M = len(test_set)
# probability = 1
# probability_unk = 1
#
# #%% pre-calculated probabilities
# bigram_probabilities = {('i', 'am'): 1.0, ('am', 'happy'): 0.5, ('happy', 'because'): 1.0, ('because', 'i'): 1.0, ('am', 'learning'): 0.5, ('learning', '.'): 1.0}
# bigram_probabilities_unk = {('i', 'am'): 1.0, ('am', '<UNK>'): 1.0, ('<UNK>', '<UNK>'): 0.5, ('<UNK>', 'i'): 0.25}
#
# #%% Calculate bigram probabiliites
# for i in range(len(test_set) - 2 + 1):
#     bigram = tuple(test_set[i:i+2])
#     probability *= bigram_probabilities[bigram]
#
#     bigram_unk = tuple(test_set_unk[i:i+2])
#     probability_unk *= bigram_probabilities_unk[bigram_unk]
#
#
# #%% Perplexity
# perplexity = probability ** (-1/M)
# perplexity_unk = probability_unk ** (-1/M)
#
#
# #%%
# trigram_probabilities = {('i', 'am', 'happy') : 2}
# bigram_probabilities = {( 'am', 'happy') : 10}
# vocabulary_size = 5
# k = 1
#
# #%%
#
# probability_known_trigram = add_k_smooting_probability(k, vocabulary_size, trigram_probabilities[('i', 'am', 'happy')],
#                            bigram_probabilities[( 'am', 'happy')])
#
# probability_unknown_trigram = add_k_smooting_probability(k, vocabulary_size, 0, 0)
# #%%
#
# print(f"probability_known_trigram: {probability_known_trigram}")
# print(f"probability_unknown_trigram: {probability_unknown_trigram}")
#
# #%% Backoff
# trigram_probabilities = {('i', 'am', 'happy') : 0}
# bigram_probabilities = {( 'am', 'happy') : 0.3}
# unigram_probabilities = {'happy' : 0.4}
#
# # this is the input trigram we need to estimate
# trigram = ('are', 'you', 'happy')
#
# # find the last bigram and unigram of the input
# bigram = trigram[1:3]
# unigram = trigram[2]
# print(f"besides the trigram {trigram} we also use bigram {bigram} and unigram ({unigram})\n")
#
# #%%
#
# lambda_factor = 0.5
# probability_hat_trigram = 0
#
# if trigram not in trigram_probabilities or trigram_probabilities[trigram] == 0:
#     print(f"Prob for {trigram} not found :-(")
#
#     if bigram not in bigram_probabilities or bigram_probabilities[bigram] == 0:
#         print(f"Prob for {bigram} not found :-(")
#
#         if unigram in unigram_probabilities:
#             print(f"Prob for {unigram} found\n")
#             probability_hat_trigram = lambda_factor * lambda_factor * unigram_probabilities[unigram]
#         else:
#             probability_hat_trigram = 0
#
#     else:
#         probability_hat_trigram = lambda_factor * bigram_probabilities[bigram]
# else:
#     probability_hat_trigram = trigram_probability[trigram]
#
# #%%
# print(f"probability for trigram {trigram} estimated as {probability_hat_trigram}")
#
#
# #%% Interpolation
# bigram_probabilities = {( 'am', 'happy') : 0.3}
# trigram_probabilities = {('i', 'am', 'happy') : 0.15}
# unigram_probabilities = {'happy' : 0.4}
#
#
# #%% weights
# lambda_1 = 0.8
# lambda_2 = 0.15
# lambda_3 = 0.05
#
# #%%
# trigram = ('i', 'am', 'happy')
# bigram = trigram[1:3]
# unigram = trigram[2]
#
# #%%
# print(f"besides the trigram {trigram} we also use bigram {bigram} and unigram ({unigram})\n")
#
# #%%
# probability_hat_trigram = lambda_1 * trigram_probabilities[trigram] \
#                         + lambda_2 *  bigram_probabilities[bigram] \
#                         + lambda_3 * unigram_probabilities[unigram]
#
# #%%
# print(f"estimated probability of the input trigram {trigram} is {probability_hat_trigram}")
#

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
    """Convert a sentence to a list n-grams order n"""
    assert ngram_order <= len(tokenized_sentence)
    ngram = []
    for i in range(len(tokenized_sentence) - ngram_order + 1):
        ngram.append(tokenized_sentence[i:i+ngram_order])
    return ngram


#%%
def preprocess_ngram_sentence(tokenized_sentence, ngram_order=2, start_token="<s>", end_token="</s>"):
    """ Add initial and end of sentnece tags"""
    return [start_token] * (ngram_order-1) + tokenized_sentence + [end_token]

#%%
def pretty_print_ngram(tokenized_sentence, ngram_order=2, normalize=True):
    """ Pretty pring a list of ngrams"""
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
    """Apply k smoothing"""
    numerator = n_gram_count + k
    denominator = n_gram_prefix_count + k * vocabulary_size
    return numerator/denominator

#%% Language model evaluation
def train_validation_test_split(data, train_percent, validation_percent, seed=87, shuffle=True):
    """
    Splits the input data to train/validation/test according to the percentage

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
    """Helper to siply split a sentence"""
    return data.strip().split('\n')

#%%
def tokenize_sentences(sentences):
    """Convert a sentence to  list of NLP tokens"""
    return [ re.findall(r"[\w']+|[.,!?;<=>]", s.lower()) for s in sentences ]

#%% More "professional"
def split_to_sentences_nltk(data):
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sentence_detector.tokenize(data.strip())

def tokenize_sentences_nltk(sentences):
    """Convert a sentence to  list of NLP tokens """
    # Missing better handling of tweeter tags, user marks and symbols
    return [nltk.word_tokenize(s.lower()) for s in sentences]

#%%
def get_tokenized_data(data):
    """Convert raw data into tokenized sentences"""
    return tokenize_sentences(split_to_sentences(data))

#%%
def count_words(data):
    """Count words in a tokenized list of sentences"""
    return Counter([ word for sentence in data for word in sentence])

#%%
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """Filter a frequency dictionay to ocurrencer above or equal count_threshold"""
    word_frequencies = count_words(tokenized_sentences)
    word_frequencies_filtered = dict(filter(lambda x: x[1] >= count_threshold, word_frequencies.items()))
    return list(word_frequencies_filtered.keys())

#%%
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """Replace OOV words by unknown token"""
    vocabulary = set(vocabulary)
    return [[ word if word in vocabulary else unknown_token for word in sentences] for sentences in tokenized_sentences ]

#%%
def preprocess_data(train_data, test_data, count_threshold):
    """Filter train/test data based on frequency threshold"""
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_pp = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_pp = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_pp, test_data_pp, vocabulary

#%% # Language Model
def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    """Count all n-grams in dataframe
    Input:
        data: list of lists of words
        n: ngram order
    Returns:
        Dicionary that maps ngram to frequency
    """
    n_grams = {}

    for sentence in data:
        sentence = (n) * [start_token] + sentence + [end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i:i+n])
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1

    return n_grams

#%%
def estimate_probability(word,
                         previous_n_gram,
                         n_gram_counts,
                         n_plus1_gram_counts,
                         vocabulary_size,
                         k=1.0):
    """Estimate probabilities of a next word using the n-gram counts
    with k-Smoothing

    Input:
        word: next word
        previous_n_gram: Dicitonar of counts of n-grams
        n_plus1_gram_counts: dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in vocabulary
        k: positive constant, smoothing parameter
    """
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + vocabulary_size * k
    n_plus1_gram = tuple(list(previous_n_gram) + [word])
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    return probability

#%%
def estimate_probabilities(previous_n_gram,
                             n_gram_counts,
                             n_plus1_gram_counts,
                             vocabulary,
                             k=1.0):
    """Estimates probabilities of a list of words given a previous_ngram"""
    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + [ "<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word,
                                           previous_n_gram,
                                           n_gram_counts,
                                           n_plus1_gram_counts,
                                           vocabulary_size,
                                           k=k)
        probabilities[word] = probability

    return probabilities

#%%
def make_count_matrix(n_plus1_gram_counts, vocabulary):
    """Add <e> <unk> to the vocabulary, <s> is omitted
    since it should not appear as the neext word_length"""
    vocabulary = vocabulary + ["<e>", "<unk>" ]

    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)

    n_grams = list(set(n_grams))

    row_index = { n_gram:i for i, n_gram in enumerate(n_grams)}
    col_index = { word: j for j, word in enumerate(vocabulary)}

    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count

    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)

    return count_matrix

#%%
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    """Builder to a probability matrix"""
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

#%%
def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """Calculate perplexity for a list of sentences

    Input:
        Sentence: list of strings
        n_gram_counts: Dictionarry of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: positive smnoothing

    Returns:
        Perplexity score
    """
    n = len(list(n_gram_counts.keys())[0])
    sent = n * ["<s>"]  + sentence + ["<e>"]
    sent = tuple(sent)
    N = len(sent)
    product_pi = 1.0

    for t in range(n, N):
        word = sent[t]
        n_gram = sent[t-n:t]
        probability =  estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        product_pi *= probability

    perplexity = product_pi ** (-1/N)

    return perplexity

#%%
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """Get suggestion for a next words

    Input:
    previous_tokens: The sentence you input where each token is a word. Must have length > n
    n_gram_counts: Dictionary of counts of n-grams
    n_plus1_gram_counts: Dictionary of counts of (n+1) -grams
    vocabulary: list of words
    k: smoothing thermal
    start_with: If not none, filter suggest words starting with this string

    Returns:
    tuple: (string of most likely next word, probability)
    """
    n = len(list(n_gram_counts.keys())[0])

    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts,
                                           n_plus1_gram_counts,
                                           vocabulary,
                                           k)

    suggestion = None
    max_prob = 0

    for word, prob in probabilities.items():

        if start_with is not None:
            if not word.startswith(start_with):
                continue

        if prob > max_prob:
            max_prob = prob
            suggestion = word

    return suggestion, max_prob

#%%
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    """Return a list of suggestions"""
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions


# ####################################################
# #%% TODO: FUNCTION FOR THIS -->  Backoff Techniqu
# ####################################################

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
# ####################################################
# #%% TODO: FUNCTION FOR THIS --> Interpolation technique
# ####################################################
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
# #%% \sum_lambda == 1
# probability_hat_trigram = lambda_1 * trigram_probabilities[trigram] \
#                         + lambda_2 *  bigram_probabilities[bigram] \
#                         + lambda_3 * unigram_probabilities[unigram]
#
# #%%
# print(f"estimated probability of the input trigram {trigram} is {probability_hat_trigram}")
#


#%% Preprocessing
with open("data/en_US.twitter.txt", "r") as fd:
    data = fd.read()

#%%
train_data, _, test_data = train_validation_test_split(get_tokenized_data(data), 80, 0, seed=87, shuffle=True)

#%%
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, minimum_freq)

#%% TEST SUGGESTION MECHANISM
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
unigram_counts  = count_n_grams(sentences, 1)
bigram_counts   = count_n_grams(sentences, 2)
trigram_counts  = count_n_grams(sentences, 3)
quadgram_counts = count_n_grams(sentences, 4)
qintgram_counts = count_n_grams(sentences, 5)

n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]

#%% Test 1: Bigrams
previous_tokens = ["i", "like"]
tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")

#%% Test2: Bigrams + starts with
tmp_starts_with = 'c'
tmp_suggest2 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0, start_with=tmp_starts_with)
print(f"The previous words are 'i like', the suggestion must start with `{tmp_starts_with}`\n\tand the suggested word is `{tmp_suggest2[0]}` with a probability of {tmp_suggest2[1]:.4f}")

#%% Test3: Multiple n-grams
previous_tokens = ["i", "like"]
tmp_suggest3 = get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)
print(f"The previous words are 'i like', the suggestions are:")
display(tmp_suggest3)

#%% Test4: Variable length
n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)

#%%
previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)
print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest4)

#%%
previous_tokens = ["i", "want", "to", "go"]
tmp_suggest5 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)
print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest5)

#%%
previous_tokens = ["hey", "how", "are"]
tmp_suggest6 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)
print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest6)

#%%
previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest7 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)
print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest7)

#%%
previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest8 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with="d")
print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest8)

#!/usr/bin/env python

#%%
import math
import string
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict

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
def pmatrix(matrix, index, cols=None):
    if cols is None: cols = index
    print(pd.DataFrame(matrix, index=index, columns=cols))

#%%
def create_dictionaries(training_corpus, vocab):
    """ Build the count dictionaries
    Input:
        training_corpus: A taggeed corpus "WORD\tTAG\n" per line
        vocab: A dicionary where keys are words and value its index
    Output:
        emission_counts: Counts of (tag, word)
        transition_counts: Counts of (tag, tag)
        tag_counts: Counts of tag
    """
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    # Start of line
    prev_tag = "--s--"

    # Line count
    for i, word_tag in enumerate(training_corpus):
        if i % 50000 == 0:
            print(f"Word count = {i}")

        word, tag = get_word_tag(word_tag, vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
        #print(f"{i:5d} -- ptag: {prev_tag} -- tag: {tag} -- word: {word}")

    return emission_counts, transition_counts, tag_counts

#%%
def predict_pos_mm(prep, y, emission_counts, vocab, states):
    num_correct = 0

    # (tag,word)
    all_words = set(emission_counts.keys())

    total = len(y)
    for word, y_tup in zip(prep, y):
        y_tup_l = y_tup.split()

        if len(y_tup_l) == 2:
            true_label = y_tup_l[1]
        else:
            # y tuple do not countain (word, pos). Need to check why...
            continue

        count_final = 0
        pos_final = ''

        if word in vocab:
            for pos in states:
                key = (pos, word)

                if key in emission_counts:
                    count = emission_counts[key]
                    if count > count_final:
                        count_final = count
                        pos_final = pos
            if pos_final == true_label:
                num_correct += 1

    accuracy = num_correct / total
    return accuracy

#%%
def create_transition_matrix(alpha, tag_counts, transition_counts):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags, num_tags))

    trans_keys = set(transition_counts.keys())

    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i], all_tags[j])
            if key in transition_counts:
                count = transition_counts[key]
            count_prev_tag = tag_counts[all_tags[i]]

            A[i,j] = (count + alpha)/(count_prev_tag + (alpha * num_tags))

    return A

#%%
def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab)
    #all_words = list(vocab.keys())
    all_words = vocab
    B = np.zeros((num_tags, num_words))
    emis_keys = set(list(emission_counts.keys()))

    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key = (all_tags[i], all_words[j])
            if key in emission_counts:
                count = emission_counts[(key)]
            count_tag = tag_counts[all_tags[i]]

            B[i,j] = (count + alpha)/(count_tag + alpha * num_words)

    return B

#%% Viterbi algorithm
def viterbi_initialize(states, tag_counts, A, B, corpus, vocab):
    """Initialize the helpe matrices to viterbi algorithms.
    Input:
        states: a list of PoS
        tag_counts: dict mapping PoS to count.
        A: Transmission matrix dim: (num_tags, num_tags)
        B: Emission matrix dim: (num_tags, len(vocab))
        corpus: List of words: format: WORD\tPOS\n
        vocab: a dictionary with keys as words and index as id.
    Output:
        best_probs: matrix (num_tags, len(corpus)) of floats
        best_paths: matrix (num_tags, len(corpus)) of ints
    """
    num_tags = len(tag_counts)
    C = np.zeros((num_tags, len(corpus)), dtype=np.float)
    D = np.zeros((num_tags, len(corpus)), dtype=np.int)

    s_idx = states.index('--s--')
    for i in range(len(states)):
        if A[s_idx, i] == 0:
            C[i, 0] = np.float('-inf')
        else:
            C[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])

    return C, D

#%%
def viterbi_forward(A, B, corpus, best_probs, best_paths, vocab):
    """A: Transition matrix np.array dim(TAGS, TAGS)
       B: Emission Matrix np.array dim(TAGS, #VOCAB)
       corpus: list of preprocessed word to fit
       best_props: np.array (matrix C) dim(TAGS, #CORPUS)
       best_paths: np.array (matrix D) dim(TAGS, #CORPUS)
       vocab: list of valid words
    """
    num_tags = best_probs.shape[0]
    C = best_probs
    D = best_paths

    for i in range(1, len(corpus)):
        if i % 5000 == 0:
            print(f"words processed: {i:>8}")

        # For each unique PoS tag that word can be
        for j in range(A.shape[1]):
            C_i = np.float('-inf')
            D_i = None
            # For each PoS tag that previous word can be
            for k in range(A.shape[0]):
                cindex = vocab[prep[i]]
                b = math.log(B[j, cindex])
                a = math.log(A[k, j])
                c = C[k,i-1]
                prob = c + a + b
                if prob > C_i:
                    C_i = prob
                    D_i = k
            C[j, i] = C_i
            D[j, i] = D_i

    return C, D

#%%
def viterbi_backward(best_probs, best_paths, corpus, states):
    """Return the best path"""
    C = best_probs
    D = best_paths
    m = best_paths.shape[1]
    z = [None] * m
    num_tags = best_probs.shape[0]
    best_prob_for_last_word = float('-inf')
    pred = [None] * m

    s = np.argmax(C[:, -1])
    pred[m - 1] = states[s]
    z[m - 1] = D[s, -1]

    for i in range(m-1, 0, -1):
        pred[i-1] = states[z[i]]
        z[i-1] = D[z[i], i-1]

    return pred

#%%
def compute_accuracy(y_predicted, y_labels):
    num_correct = 0
    total = 0

    for p, y in zip(y_predicted, y_labels):
        word_tag_tuple = y.split()

        if len(word_tag_tuple) != 2:
            continue

        word, tag = word_tag_tuple

        if p == tag:
            num_correct += 1

        total += 1

    return num_correct/total

#%%
with open("data/WSJ_02-21.pos", "r") as f:
    training_corpus = f.readlines()

#%%
with open("data/hmm_vocab.txt", "r") as f:
    voc_l = f.read().split('\n')

#%%
vocab = {}
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i

#%%
with open("data/WSJ_24.pos", "r") as f:
    y = f.readlines()

#%%
_, prep = preprocess(vocab, "data/test.words")

#%%
emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)

#%%
states = sorted(tag_counts.keys())

#%%
accuracy_predict_pos = predict_pos_mm(prep, y, emission_counts, vocab, states)
print(f"Accuracy of prediction using predict_pos_mm is {accuracy_predict_pos:.4f}")

#%%
alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)

#%%
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))

#%%
best_probs, best_paths = viterbi_initialize(states, tag_counts, A, B, prep, vocab)

#%%
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)

#%%
pred = viterbi_backward(best_probs, best_paths, prep, states)

#%%
print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")


### These are tests used to test the functions
### Need to rewrite them to be more helpful.
###
### #%% TESTES
### # Testing your function
### print("View a subset of transition matrix A")
### A_sub = pd.DataFrame(A[30:35,30:35], index=states[30:35], columns = states[30:35] )
### print(A_sub)
###
### #%%
### # creating your emission probability matrix. this takes a few minutes to run.
###
### print(f"View Matrix position at row 0, column 0: {B[0,0]:.9f}")
### print(f"View Matrix position at row 3, column 1: {B[3,1]:.9f}")
###
### # Try viewing emissions for a few words in a sample dataframe
### cidx  = ['725','adroitly','engineers', 'promoted', 'synergy']
###
### # Get the integer ID for each word
### cols = [vocab[a] for a in cidx]
###
### # Choose POS tags to show in a sample dataframe
### rvals =['CD','NN','NNS', 'VB','RB','RP']
###
### # For each POS tag, get the row number from the 'states' list
### rows = [states.index(a) for a in rvals]
###
### # Get the emissions for the sample of words, and the sample of POS tags
### B_sub = pd.DataFrame(B[np.ix_(rows,cols)], index=rvals, columns = cidx )
### print(B_sub)
###
### #%% Test viterbi initialization
### print(f"best_probs[0,0]: {best_probs[0,0]:.4f}")
### print(f"best_paths[2,3]: {best_paths[2,3]:.4f}")
###
### #% Teest viterbi forward
### print(f"A at row 0, col 0: {A[0,0]:.9f}")
### print(f"A at row 3, col 1: {A[3,1]:.4f}")
###
### #%%
### m=len(pred)
### print('The prediction for pred[-7:m-1] is: \n', prep[-7:m-1], "\n", pred[-7:m-1], "\n")
### print('The prediction for pred[0:8] is: \n', pred[0:7], "\n", prep[0:7])
###
### #%%
### print('The third word is:', prep[3])
### print('Your prediction is:', pred[3])
### print('Your corresponding label y is: ', y[3])
###

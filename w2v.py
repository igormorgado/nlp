#!/usr/bin/env bash

#%%
import re
import emoji
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize

#%% Convert punctuation into dots
def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data.lower())
    data = [ w for w in data if w.isalpha() or w == '.' or emoji.get_emoji_regexp().search(w) ]
    return data

#%%
def get_windows(sentence, C=2):
    i = C
    while i < len(sentence) - C:
        center_word = sentence[i]
        context_words = tuple(sentence[(i-C):i] + sentence[(i+1):(i+C+1)])
        yield (context_words, center_word)
        i += 1

#%%
def get_dict(data):
    """
    Input:
        data: = list of words

    Output:
        word2ind = dict of word to index
        ind2word = dict of index to word_tokenize
    """

    vocab = sorted(list(set(data)))
    word2ind = {}
    ind2word = {}
    for i,w in enumerate(vocab):
        word2ind[w] = i
        ind2word[i] = w
    return word2ind, ind2word

#%%
def wrd2hot(word, word2ind):
    # Why not count word2ind here?
    V = len(word2ind)
    ohv = np.zeros(V)
    # IF word do not exist? What we should do?
    ohv[word2ind[word]] = 1
    return ohv

#%%
def ctx2hot(cwords, word2ind):
    """Convert a list of words (often context words) into a single mean vector."""
    wordvectors = [ wrd2hot(cw, word2ind) for cw in cwords ]
    wordvecmean = np.mean(wordvectors, axis=0)
    return wordvecmean

#%%
def hot2word(ohv, ind2word):
    word = ind2word[np.argmax(ohv)]
    return word

#%%
def get_training_sample(sentence, word2ind, C=2):
    for context_words, center_word in get_windows(sentence, C):
        X = ctx2hot(context_words, word2ind)
        Y = wrd2hot(center_word, word2ind)
        yield (X, Y)

#%%
def relu(Z):
    return np.maximum(0, Z)

#%%
def softmax(Z):
    ez = np.exp(Z)
    sum_ez = np.sum(ez)
    return ez / sum_ez

#%%
def Wxb(W, x, b):
    return W @ x + b

#%%
def initialize_cbow_weights(N, V):
    """ Returns initialized weights and biases
    Input:
        N: Word eembedding size (hyper parameter)
        V: Vocabulary size

    Output:
        W1: Dim(NxV)
        b1: Dim(Nx1)
        W2: Dim(VxN)
        b2: Dim(Vx1)
    """
    W1 = np.random.random((N, V))
    b1 = np.random.random((N, 1))
    W2 = np.random.random((V, N))
    b2 = np.random.random((V, 1))
    return W1, b1, W2, b2

#%% Loss function
def cross_entropy_loss(ypred, y):
    loss = np.sum(-np.log(ypred) * y)
    return loss



#%%
# corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'
corpus = 'I am happy because I am learning'
sentence = tokenize(corpus)
word2ind, ind2word = get_dict(sentence)
C = 2
V = len(word2ind)
N = 3
alpha = 0.03

#%% Fetch samples
training_samples = get_training_sample(sentence, word2ind, 2)

#%% Initialize weights
# W1, b1, W2, b2 = initialize_cbow_weights(N, V)
W1 = np.array([[ 0.41687358,  0.08854191, -0.23495225,  0.28320538,  0.41800106],
               [ 0.32735501,  0.22795148, -0.23951958,  0.4117634 , -0.23924344],
               [ 0.26637602, -0.23846886, -0.37770863, -0.11399446,  0.34008124]])

W2 = np.array([[-0.22182064, -0.43008631,  0.13310965],
               [ 0.08476603,  0.08123194,  0.1772054 ],
               [ 0.1871551 , -0.06107263, -0.1790735 ],
               [ 0.07055222, -0.02015138,  0.36107434],
               [ 0.33480474, -0.39423389, -0.43959196]])

b1 = np.array([[ 0.09688219],
               [ 0.29239497],
               [-0.27364426]])

b2 = np.array([[ 0.0352008 ],
               [-0.36393384],
               [-0.12775555],
               [-0.34802326],
               [-0.07017815]])

#%% Compute word embeddings
for x, y in training_samples:
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    #%% Compute forward pass
    z1 = W1 @ x + b1
    h = relu(z1)
    z2 = W2 @ h + b2
    yhat = softmax(z2)

    #%% Back propagations
    grad_b2 = yhat - y
    grad_W2 = (yhat - y) @ h.T
    grad_b1 = relu(W2.T @ (yhat - y))
    grad_W1 = relu( W2.T @ (yhat - y)) @ x.T

    #%% Gradient descent
    W1 -= alpha * grad_W1
    W2 -= alpha * grad_W2
    b1 -= alpha * grad_b1
    b2 -= alpha * grad_b2

#%% extract word embeddings

W3 = (W1+W2.T)/2

#%% Tester
for word in word2ind:
    word_embedding_vector = W3[:, word2ind[word]]
    print(f'{word:10s}: {word_embedding_vector}')


#!/usr/bin/env bash

#%%
import re
import emoji
import numpy as np
import nltk
import string
from scipy import linalg
from collections import defaultdict
from nltk.tokenize import word_tokenize
from pca import compute_pca

#%% Convert punctuation into dots
def tokenize(corpus):
    #data = re.sub(r'[,!?;-]+', '.', corpus)
    data = re.sub(r'[,!?;-]', '.', corpus)
    data = nltk.word_tokenize(data)
    #data = [ w for w in data if w.isalpha() or w == '.' or emoji.get_emoji_regexp().search(w) ]
    data = [ w.lower() for w in data if w.isalpha() or w == '.' ]
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
    sum_ez = np.sum(ez, axis=0, keepdims=1)
    return ez / sum_ez

#%%
def initialize_model(N, V, random_seed=1):
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
    np.random.seed(random_seed)
    W1 = np.random.rand(N, V)
    W2 = np.random.rand(V, N)
    b1 = np.random.rand(N, 1)
    b2 = np.random.rand(V, 1)
    return W1, W2, b1, b2

#%% Loss function
def cross_entropy_loss(ypred, y):
    loss = np.sum(-np.log(ypred) * y)
    return loss

#%%
def forward_prop(x, W1, W2, b1, b2):
    z1 = W1 @ x + b1
    h = relu(z1)
    z = W2 @ h + b2
    return z, h

#%%
def compute_cost(y, yhat, batch_size):
    logprobs = np.multiply(np.log(yhat),y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

#%%
def sigmoid(z):
    # sigmoid function
    return 1.0/(1.0+np.exp(-z))

#%%
def get_idx(words, word2ind):
    idx = []
    for word in words:
        idx = idx + [word2ind[word]]
    return idx

#%%
def pack_idx_with_frequency(context_words, word2ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed

#%%
def get_vectors(data, word2ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2ind[center_word]] = 1
        context_words = data[(i - C):i] + data[(i+1):(i+C+1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2ind):
            x[idx] = freq/num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            print('i is being set to 0')
            i = 0

#%%
def get_batches(data, word2ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch = []

#%%
def back_prop(W2, x, y, yhat, h, batch_size):
    ydiff = yhat - y
    l1 = W2.T @ ydiff
    l1 = relu(l1)
    grad_W1 = (l1 @ x.T)/batch_size
    grad_W2 = (ydiff @ h.T)/batch_size
    grad_b1 = np.sum(l1, axis=1, keepdims=True)/batch_size
    grad_b2 = np.sum(ydiff, axis=1, keepdims=True)/batch_size
    return grad_W1, grad_W2, grad_b1, grad_b2

#%%
def gradient_descent(data, word2ind, N, V, num_iters, alpha=0.03, batch_size=128):

    '''
    This is the gradient_descent function

      Inputs:
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector
        V:         dimension of vocabulary
        num_iters: number of iterations
     Outputs:
        W1, W2, b1, b2:  updated matrices and biases

    '''
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=282)
    iters = 0
    C = 2
    for x, y in get_batches(data, word2ind, V, C, batch_size):
        z, h = forward_prop(x, W1, W2, b1, b2)
        yhat = softmax(z)
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(W2, x, y, yhat, h, batch_size)

        # Update weights and biases
        W1 -= alpha * grad_W1
        W2 -= alpha * grad_W2
        b1 -= alpha * grad_b1
        b2 -= alpha * grad_b2

        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2

if __name__ == '__main__':
#%% Hyper params
    C = 2
    N = 50
    num_iters = 150

#%%
    with open('data/shakespeare.txt', 'r') as fd:
        data = fd.read()

#%%
    data = tokenize(data)

#%%
    fdist = nltk.FreqDist(word for word in data)

#%%
    word2ind, ind2word = get_dict(data)
    V = len(word2ind)

#%%
    print("Call gradient_descent")
    W1, W2, b1, b2 = gradient_descent(data, word2ind, N, V, num_iters)

#%% visualizing the word vectors here
    from matplotlib import pyplot
    words = ['king', 'queen','lord','man', 'woman','dog','wolf',
             'rich','happy','sad']

    embs = (W1.T + W2)/2.0

    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    idx = [word2ind[word] for word in words]
    X = embs[idx, :]
    print(X.shape, idx)  # X.shape:  Number of words of dimension N each

    result = compute_pca(X, 2)
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

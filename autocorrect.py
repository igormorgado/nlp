#!/usr/bin/env python

#%%
from collections import Counter
import re
import numpy as np
import pandas as pd

#%%
def process_data(filename):
    with open(filename, 'r') as fd:
        words = [word for line in fd.readlines() for word in re.findall(r'\w+', line.lower().strip())]
    return words

#%%
def get_count(wordlist):
    return Counter(wordlist)

#%%
def get_probs(worddict):
    total = sum(worddict.values())
    probs = {k: v / total for k, v in worddict.items()}
    return probs

#%%
def split_word(word):
    return [(word[:i], word[i:]) for i in range(len(word)+1)]

#%%
def delete_letter(word, verbose=False):
    splits = split_word(word)
    deletes = [L + R[1:] for L, R in splits if R]
    if verbose: print(f"input word {word}, \nsplits = {splits}, \ndeletes = {deletes}")
    return deletes

#%%
def switch_letter(word, verbose=False):
    splits = split_word(word)
    switchs = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    if verbose: print(f"Input word = {word} \nsplits = {splits} \nswitchs = {switchs}")
    return switchs

#%%
def replace_letter(word, verbose=False):
    splits = split_word(word)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replaces = [L + N + R[1:] for L, R in splits for N in letters if (R and N != R[0])]
    if verbose: print(f"input word {word}, \nsplits = {splits}, \nreplaces = {replaces}")
    return replaces

#%%
def insert_letter(word, verbose=False):
    splits = split_word(word)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    inserts = [L + N + R[0:] for L, R in splits for N in letters]
    if verbose: print(f"input word {word}, \nsplits = {splits}, \ninserts = {inserts}")
    return inserts

#%%
def edit_one_letter(word, allow_switches=True):
    if allow_switches:
        edit_one_set = set(delete_letter(word) + insert_letter(word) + switch_letter(word) + replace_letter(word) )
    else:
        edit_one_set = set(delete_letter(word) + insert_letter(word) + replace_letter(word) )

    return edit_one_set

#%%
def edit_two_letter(word, allow_switches=True):
    edit_two_set = set([v for u in edit_one_letter(word, allow_switches) for v in edit_one_letter(u, allow_switches)])
    return edit_two_set

#%%
def get_corrections(word, probs, vocab, n=2, verbose=False):
    suggestions = []
    n_best = []

    if word in vocab:
        suggestions.append(word)
        n_best.append((word, probs.get(word, 0)))

    new_suggestions = edit_one_letter(word).intersection(vocab).difference(set(suggestions))
    suggestions.extend(new_suggestions)
    new_suggestions = [(w, probs.get(w, 0)) for w in new_suggestions]
    new_suggestions = sorted(new_suggestions, key=lambda x: x[1], reverse=True)
    n_best.extend(new_suggestions)

    if len(n_best) < n:
        new_suggestions = edit_two_letter(word).intersection(vocab).difference(suggestions)
        suggestions.extend(new_suggestions)
        new_suggestions = [(w, probs.get(w, 0)) for w in new_suggestions]
        new_suggestions = sorted(new_suggestions, key=lambda x: x[1], reverse=True)
        n_best.extend(new_suggestions)

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best[:n]

#%%
def min_edit_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2):
    '''
    Input:
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you want to end with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target
    '''
    # use deletion and insert cost as  1
    m = len(source)
    n = len(target)
    D = np.zeros((m+1, n+1), dtype=int)
    for row in range(1,m+1):
        D[row,0] = row

    for col in range(1,n+1):
        D[0,col] = col

    for row in range(1,m+1):
        for col in range(1,n+1):
            r_cost = rep_cost
            if source[row-1] == target[col-1]:
                r_cost = 0

            D[row,col] = min(D[row-1, col]   + del_cost,
                             D[row-1, col-1] + r_cost,
                             D[row,   col-1] + ins_cost)

    med = D[row, col]
    return D, med

#%% TESTS!
words = process_data('data/shakespeare.txt')
vocab = set(words)

#%% Tests
print(f"The first ten words in the text are: \n{words[0:10]}")
print(f"There are {len(vocab)} unique words in the vocabulary.")

#%%
word_count_dict = get_count(words)
print(f"There are {len(word_count_dict)} key values pairs")
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")

#%%
probs = get_probs(word_count_dict)
print(f"Length of probs is {len(probs)}")
print(f"P('thee') is {probs['thee']:.4f}")

#%%
my_word = 'dys'
tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose=True) # keep verbose=True
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")

print(f"data type of corrections {type(tmp_corrections)}")

#%%
source =  'eer'
target = 'near'
matrix, min_edits = min_edit_distance(source, target)
print("minimum edits: ",min_edits, "\n")
idx = list(source)
idx.insert(0, '#')
cols = list(target)
cols.insert(0, '#')
df = pd.DataFrame(matrix, index=idx, columns= cols)
print(df)

#%%
source = "eer"
targets = edit_one_letter(source, allow_switches = False)
for t in targets:
    _, min_edits = min_edit_distance(source, t, 1, 1, 1)
    if min_edits != 1:
        print(source, t, min_edits)

#%%
source = "eer"
targets = edit_two_letter(source,allow_switches = False)
for t in targets:
    _, min_edits = min_edit_distance(source, t,1,1,1)
    if min_edits != 2 and min_edits != 1: print(source, t, min_edits)

#!/usr/bin/env python

#%%
import nltk
from gensim.models import KeyedVectors

#%%
embeddings = KeyedVectors.load_word2vec_format('/data/igor/GoogleNews-vectors-negative300.bin', binary=True)

#%%

word = "king"
widx = embeddings.key_to_index[word]
wcnt = embeddings.get_vecattr(word, "count")
#%%
king_vec = embeddings.get_vector("king")
man_vec = embeddings.get_vector("man")
woman_vec = embeddings.get_vector("woman")
queen_maybe = king_vec - man_vec + woman_vec
#%%
print(embeddings.similar_by_vector(queen_maybe))

queen_vec = embeddings.get_vector("queen")


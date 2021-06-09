#!/usr/bin/env python

#%%
import numpy as np
import pickle
from gensim.models import KeyedVectors
from nlp_helper import cosine_similarity, build_dict
from sentiment_utils import process_tweet

#%% Build vector model
def build_vector_model (src_subset, dst_subset,
                        train_dictionary, test_dictionary,
                        src_embeddings, dst_embeddings,
                        src_binary=False, dst_binary=False):

    # Read word2vec language models
    print('Reading embeddings...')
    en_embeddings = KeyedVectors.load_word2vec_format(src_embeddings, binary=src_binary)
    fr_embeddings = KeyedVectors.load_word2vec_format(dst_embeddings, binary=dst_binary)

    # Embedding model vocabulary
    en_vocab = set(en_embeddings.key_to_index.keys())
    fr_vocab = set(fr_embeddings.key_to_index.keys())

    # Read the dictionaries
    en_fr_train = build_dict(train_dictionary)
    en_fr_test = build_dict(test_dictionary)

    # Target words (en -> fr)
    # In this case we ignore multiple word meanings
    fr_words = set(en_fr_train.values())

    # Embedding subset with only words in target dict
    print('Building embeddings subset...')
    en_embeddings_subset = {}
    fr_embeddings_subset = {}

    # Build subset with train items
    for en_word, fr_word in en_fr_train.items():
        if en_word in en_vocab and fr_word in fr_vocab:
            en_embeddings_subset[en_word] = en_embeddings[en_word]
            fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]

    # And with test items
    for en_word, fr_word in en_fr_test.items():
        if en_word in en_vocab and fr_word in fr_vocab:
            en_embeddings_subset[en_word] = en_embeddings[en_word]
            fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]

    # Save embeddings subset to file
    print('Writting embeddings subset to disk...')
    pickle.dump (en_embeddings_subset, open(src_subset, 'wb'))
    print(f'    {src_subset}')
    pickle.dump (fr_embeddings_subset, open(dst_subset, 'wb'))
    print(f'    {dst_subset}')

    return

#%% Build matrices from embeddings
def build_matrices(words, src_vecs, dst_vecs):
    src_vocab = set(src_vecs.keys())
    dst_vocab = set(dst_vecs.keys())
    dst_words = set(words.values())

    X = []
    Y = []
    for src_word, dst_word in words.items():
        if src_word in src_vocab and dst_word in dst_vocab:
            src_vec = src_vecs[src_word]
            dst_vec = dst_vecs[dst_word]
            X.append(src_vec)
            Y.append(dst_vec)

    X = np.array(X)
    Y = np.array(Y)

    assert(X.shape == Y.shape)
    return X, Y

#%% Train the weights
def compute_loss(X, Y, R):
    # Every source entry exist in label
    assert(X.shape[0] == Y.shape[0])
    # X can be multiplied by R
    assert(X.shape[1] == R.shape[0])
    # Result of XR has same dimensions as Y
    assert(Y.shape[1] == R.shape[1])

    m = X.shape[0]

    diff = X @ R - Y
    # Frobenius norm
    diff = np.square(diff)
    total_sum = np.sum(diff)
    # Average
    loss = total_sum/m
    return loss

#%% Loss gradient
def compute_gradient(X, Y, R):
    # Every source entry exist in label
    assert(X.shape[0] == Y.shape[0])
    # X can be multiplied by R
    assert(X.shape[1] == R.shape[0])
    # Result of XR has same dimensions as Y
    assert(Y.shape[1] == R.shape[1])

    m = X.shape[0]
    gradient = (X.T @ ((X @ R) - Y)) * (2 / m)
    return gradient

#%% Find transformation
def compute_transformation(X, Y, epochs=1000, learning_rate=3e-4, seed=129):
    """ Find R transformation matrix that fit X into Y"""
    np.random.seed(seed)

    # Initialize a random matrix.
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(epochs):
        if i % 25 == 0:
            print(f"{i}: loss = {compute_loss(X, Y, R):.4f}")

        gradient = compute_gradient(X, Y, R)
        R -= learning_rate * gradient

    return R

#%% Naive knn
def knn_naive(v, candidates, k=1):
    """ Find the 'k'-nearest neighbour of 'v' from 'candidates'"""
    assert(v.shape[0] == candidates.shape[1])
    similarity = list(map(lambda x: cosine_similarity(v, x), candidates))
    sorted_ids = np.argsort(similarity)[::-1]
    k_idx = sorted_ids[:k]
    return k_idx

#%% Accuracy evaluation
def accuracy_evaluation(X, Y, R):
    """Evaluate the accuracy from X embeddings to Y embeddings using R transformation"""
    pred = X @ R
    hits = 0
    for i, p in enumerate(pred):
        pred_idx = knn_naive(p, Y)
        if pred_idx == i:
            hits += 1

    accuracy = hits / len(pred)
    return accuracy

#%% Build document embedding
def get_document_embedding(document, embeddings):
    """Return a embedding vector associated with a document"""

    # Need to get the embedding dimension. Instead hardcoded
    doc_embedding = np.zeros(300)

    doc_processed = process_tweet(document)
    for word in doc_processed:
        doc_embedding += embeddings.get(word, 0)

    return doc_embedding

#%%
def get_document_vecs(documents, embeddings):
    """Return all embedding vectors from a list of documents"""
    indexed_vecs = {}
    document_vec = []

    # This use twice memory for embeddings, maybe a pandas dataframe?
    for i, doc in enumerate(documents):
        doc_embedding = get_document_embedding (doc, embeddings)
        indexed_vecs[i] = doc_embedding
        document_vec.append(doc_embedding)

    document_vecs = np.vstack(document_vec)

    assert(len(indexed_vecs) == document_vecs.shape[0])
    return document_vecs, indexed_vecs

#%% Compute document hash for a given set of planes
#%% This hash is uneven, in uniform concentrate on 0 and normal on 511
def vector_hash_value(document_vec, planes):
    """Each column of planes is a plane normal vector"""

    # Compute the projections of v on hypereplanes
    dotp = document_vec @ planes

    # Compute the vector position (above/below) related to hyperplanes
    signdot = np.sign(dotp)

    # Convert sign for a multiplicative term
    h = (signdot + 1)/2
    h = h.flatten()

    # Compute hash value
    hash_value = 0
    n_planes = planes.shape[1]
    for i in range(n_planes):
        hash_value += 2**i * h[i]

    return int(hash_value)

#%% Build hash table for all documents, given a set of separation hypereplanes
def make_hash_table(document_vecs, planes):
    n_planes = planes.shape[1]
    n_buckets = 2**n_planes

    # Build empty hashtables
    # Associate a hash with a document embedding (vector)
    enc_hash_table = {k: [] for k in range(n_buckets)}
    # Associate a hash with vector index in document_vecs
    idx_hash_table = {k: [] for k in range(n_buckets)}

    for i, document_vec in enumerate(document_vecs):
        h = vector_hash_value(document_vec, planes)
        enc_hash_table[h].append(document_vec)
        idx_hash_table[h].append(i)

    return enc_hash_table, idx_hash_table

#%% Build universe hash tables
def build_multiverse(document_vecs, n_dims, n_planes, n_universes):
    # Initialize the multiverse
    planes_multiverse = [np.random.normal(size=(n_dims, n_planes)) for _ in range(n_universes)]

    enc_hash_tables = []
    idx_hash_tables = []
    for universe_id in range(n_universes):
        planes = planes_multiverse[universe_id]
        enc_hash_table, idx_hash_table = make_hash_table(document_vecs, planes)
        enc_hash_tables.append(enc_hash_table)
        idx_hash_tables.append(idx_hash_table)

    return planes_multiverse, enc_hash_tables, idx_hash_tables

#%% Fast kNN
def knn_fast(doc_id, v, planes_multiverse, hash_tables, idx_tables, k=1, num_universes_to_use=5):
    vecs_to_consider_l = list()
    ids_to_consider_l = list()
    ids_to_consider_unique = set()

    for universe_id in range(num_universes_to_use):
        planes = planes_multiverse[universe_id]
        hash_value = vector_hash_value(v, planes)
        hash_table = hash_tables[universe_id]       # PASS AS ARGUMENT IS NEEDED hash_tables
        document_vectors_l = hash_table[hash_value]
        idx_table = idx_tables[universe_id]         # PASS AS ARGUMENT IS NEEDED idx_tables
        new_ids_to_consider = idx_table[hash_value]

        # TODO: Need remove the own document from search
        # THIS IS NOT WORKING :-( why?
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        for i, new_id in enumerate(new_ids_to_consider):
            document_vector_at_i = document_vectors_l[i]
            vecs_to_consider_l.append(document_vector_at_i)
            ids_to_consider_l.append(new_id)
            ids_to_consider_unique.add(new_id)

    print(f"Fast considering {len(vecs_to_consider_l)}")
    vecs_to_consider_arr = np.array(vecs_to_consider_l)
    nearest_neighbor_idx_l = knn_naive(v, vecs_to_consider_arr, k=k)
    nearest_neighbor_ids = [ids_to_consider_l[idx] for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids


#%% Build only if subset files do not exist
# TODO: Add the test
# Data needed:
# https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
#
# build_vector_model ('./data/en_embeddings_subset.p', './data/fr_embeddings_subset.p',
#                     'data/en-fr.train.txt', 'data/en-fr.test.txt',
#                     './data/GoogleNews-vectors-negative300.bin', './data/wiki.fr.vec',
#                     True)

#%% Load word subset
en_embeddings_subset = pickle.load(open('./data/en_embeddings_subset.p', 'rb'))
fr_embeddings_subset = pickle.load(open('./data/fr_embeddings_subset.p', 'rb'))

#%% Load test/train dict data
en_fr_train = build_dict('./data/en-fr.train.txt')
en_fr_test = build_dict('./data/en-fr.test.txt')

#%% Build train data
X_train, Y_train = build_matrices(en_fr_train, en_embeddings_subset, fr_embeddings_subset)

#%%
R_train = compute_transformation(X_train, Y_train, epochs=400, learning_rate=0.8)

#%% Evaluate accuracy
X_val, Y_val = build_matrices(en_fr_test, en_embeddings_subset, fr_embeddings_subset)

#%% Preditions
accuracy = accuracy_evaluation(X_val, Y_val, R_train)
print(f"Model accuracy is: {accuracy}")

#%% Locality Sensitive Hashing
from nltk.corpus import twitter_samples
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

#%%
document_vecs, indexed_vecs = get_document_vecs(all_tweets, en_embeddings_subset)

#%% Locality sensitive Hashing
N_VECS = len(all_tweets)
N_DIMS = len(indexed_vecs[1])
BUCKET_AVG_SIZE = 16

# Estimate the number of hyperplanes to split the spacecraft
N_PLANES = int(np.ceil(np.log2(N_VECS / BUCKET_AVG_SIZE)))

# Number of different splitting to surround the vector for nearest vectors.
N_UNIVERSES = 25

#%% Compute the multiverse for LSH
planes_multiverse, enc_hash_tables, idx_hash_tables = build_multiverse(document_vecs, N_DIMS, N_PLANES, N_UNIVERSES)


#%% RUN A SMALL TEST!
doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]

#%%
nearest_neighbor_ids = knn_fast(doc_id, vec_to_search, planes_multiverse, enc_hash_tables, idx_hash_tables, k=3, num_universes_to_use=5)

#%%
print(f"Nearest neighbors for document {doc_id}")
print(f"Document contents: {doc_to_search}")
print("")

# TODO: Need remove the own document from search
for neighbor_id in nearest_neighbor_ids:
    print(f"Nearest neighbor at document id {neighbor_id}")
    print(f"document contents: {all_tweets[neighbor_id]}")
    print("")


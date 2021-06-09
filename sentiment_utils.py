#%% Load Modules
import numpy as np
import re
import string
from itertools import chain
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer

#%%
def calculate_total_prob(probs):
    tot = {}
    for (word, kls), value in probs.items():
        if kls in tot:
            tot[kls] += value
        else:
            tot[kls] = value

    return tot

#%%  Helper functions for logistic regression
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    # Replace misleading smiley
    tweet = re.sub(r': \)', ':)', tweet)
    tweet = re.sub(r': \(', ':(', tweet)
    # Remove stock market tickers.
    tweet = re.sub(r'\$\w*', '', tweet)
    # Remove retweet marks
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove emails
    tweet = re.sub(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+', '', tweet)
    # Remove hash tag sign
    tweet = re.sub(r'#', '', tweet)
    # Tokenizer
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    # Stemming
    tweet_tokens = [ stemmer.stem(tt) for tt in tweet_tokens if (tt not in string.punctuation and tt not in stopwords_en) ]
    # NO STEMMING/STOPWORDS HERE #  tweet_tokens = [ tt for tt in tweet_tokens if tt not in string.punctuation ]
    #tweet_tokens = [ tt for tt in tweet_tokens if tt not in string.punctuation ]

    return tweet_tokens

#%%
def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

#%%
def build_freqs_igu(tweets, labels):
    """ Build frequencies:
    Input:
        tweets: A list of tweets
        labels: A list of labels
    Output:
        freqs: A dictionary mapping each (word, sentiment) to frequency
    """
    freqs = {}
    for kls in set(labels):
        for k, v in Counter(chain(*[process_tweet(tweet) for tweet, label in zip(tweets, labels) if label == kls ])).items():
            freqs[(k, kls)] = v
    return freqs


def extract_features(tweet, freqs):
    pp_tweet = process_tweet(tweet)
    pos, neg = 0, 0
    # Use set() to avoid repeated tags
    for word in pp_tweet:
        pos += freqs.get((word, 1), 0)
        neg += freqs.get((word, 0), 0)
    return np.array([1, pos, neg], dtype=float)

#%% Helper functions for all.


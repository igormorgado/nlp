#%% Load Modules
import numpy as np
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
from sentiment_utils import *

#%% Define some functions
def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

#%%
def gradientDescent( X, y, t, a, num_iters):
    m = X.shape[0]

    for i in range(num_iters):
        z = X @ t
        h = sigmoid(z)
        J = -(1/m) * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1-h)))
        t = t -  ((a/m) * np.dot(X.transpose(), (h-y)))

    J = float(J)
    return J, t

#%%
def extract_features(tweet, freqs):
    pp_tweet = process_tweet(tweet)
    pos, neg = 0, 0
    # Use set() to avoid repeated tags
    for word in pp_tweet:
        pos += freqs.get((word, 1), 0)
        neg += freqs.get((word, 0), 0)
    return np.array([1, pos, neg], dtype=float)

#%%
def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(x @ theta)
    return y_pred

#%%
def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    accuracy = np.sum(np.array(y_hat) == test_y.flatten())/len(y_hat)
    return accuracy

#%% Load Dataset
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#%% Split data in train and test
splitidx = 4000

train_pos  = all_positive_tweets[:splitidx]
train_neg  = all_negative_tweets[:splitidx]
train_x = np.array(train_pos + train_neg)
train_y = np.expand_dims(np.array(([1] * len(train_pos) + [0] * len(train_neg)), dtype=float), -1)

test_pos = all_positive_tweets[splitidx:]
test_neg = all_negative_tweets[splitidx:]
test_x = np.array(test_pos + test_neg)
test_y = np.expand_dims(np.array([1] * len(test_pos) + [0] * len(test_neg)), -1)

#%% SLOW:  Calculate frequencies from training data
freqs = build_freqs(train_x, train_y)


#%% SLOW:  Extract the features for all inputs
m = len(train_x)
X = np.zeros((m, 3))
for i, tweet in enumerate(train_x):
    X[i, :] = extract_features(tweet, freqs)

Y = train_y

#%% SLOW: Apply gradient descent
J, theta = gradientDescent (X, Y, np.zeros((3, 1)), 1e-9, 1500)

#%% Test accuracy
#tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
#print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

#%% Misleading tweets
# Some error analysis done for you
# print('Label Predicted Tweet')
# for x,y in zip(test_x,test_y):
#     y_hat = predict_tweet(x, freqs, theta)
#
#     if np.abs(y - (y_hat > 0.5)) > 0:
#         print('THE TWEET IS:', x)
#         print('THE PROCESSED TWEET IS:', process_tweet(x))
#         print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))

#%%

while True:
    print("Digite um 'tweet' em ingles, pra eu adivinhas como voce esta se sentindo: ")
    my_tweet = input()
    # my_tweet = "I'm happy."

    # my_tweet = my_tweet.strip()
    y_hat = predict_tweet(my_tweet, freqs, theta)[0]

    if y_hat >= 0.5:
        print(f":-)")
    else:
        print(f":-(")
    # {y_hat:.2f}
    print()
    print()


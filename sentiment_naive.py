#%% Load Modules
import numpy as np
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sentiment_utils import *

#%% naive_bayes(X, y):
def naive_bayes_train(X, y, model=None):
    """ X already preprocessed
        y are the labels"""

    if model is None:
        # Compute the word frequencies
        n_class = { 1:0, 0:0 }
        ylist = np.squeeze(y).tolist()
        for kls, entry in zip(ylist, X):
            for word in entry:
                pair = (word, kls)
                n_class[kls] +=1
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1
    else:
        freqs, n_class = model

    # Compute log prior
    count = np.bincount(y.flatten().astype(int))
    countidx = np.nonzero(count)[0]
    c = dict(zip(countidx, count[countidx]))
    logprior = np.log(c[1]) - np.log(c[0])

    # Compute the vocabulary
    vocabulary = { word for word, kls in freqs }
    V = len(vocabulary)

    # Compute probabilities with Laplacian smooth and lambdas
    probs = {}
    loglikelihood = {}
    for word in vocabulary:
        probs[(word, 0)] = np.log((freqs.get((word, 0), 0) + 1) / (n_class[0] + V))
        probs[(word, 1)] = np.log((freqs.get((word, 1), 0) + 1) / (n_class[1] + V))
        loglikelihood[word] = probs[(word,1)] - probs[(word, 0)]

    return loglikelihood, logprior, probs

#%%
def naive_bayes_extract_features(X, y, logprobs):
    Xfeatures = np.zeros((len(y),3))
    ylist = np.squeeze(y).tolist()
    for i, (tweet, kls) in enumerate(zip(X, ylist)):
        Xfeatures[i] = [np.sum([ logprobs.get((word, 1), 0) for word in tweet]),
                        np.sum([ logprobs.get((word, 0), 0) for word in tweet]),
                        kls]
    return Xfeatures

#%% Evaluate naive bayes model
def naive_bayes_predict(tweet, lambdas, logprior):
    ptweet = process_tweet(tweet)
    likelihood = sum(map(lambda x:lambdas.get(x, 0), ptweet))
    return 1 if (logprior + likelihood) > 0 else 0

#%% Test accuracy(
def naive_bayes_accuracy(X, y, lambdas, logprior):
    y_hat = np.array([ naive_bayes_predict(tweet, lambdas,logprior) for tweet in X ])
    error = np.mean(np.abs(y_hat - y.squeeze()))
    accuracy = 1 - error
    return accuracy

#%% Build the frequencies for naive bayes (also compute the word total
def naive_bayes_build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    n_class = { 1:0, 0:0 }
    for y, tweet in zip(yslist, tweets):
        for word in tweet:
            pair = (word, y)
            n_class[y] +=1
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs, n_class

#%% Tell if the word as a good or bad sentment
def get_ratio(freqs, word):
    p['positive'] = freqs.get((word,1), 0)
    p['negative'] = freqs.get((word,0), 0)
    p['ratio'] = (p['positive']+1) /  (p['negative']+1)
    return p

#%%
def get_words_by_threshold(freqs, label, threshold):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}
    for key in freqs.keys():
        word, _ = key
        pos_neg_ratio = get_ratio(freqs, word)
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
            word_list[word] = pos_neg_ratio
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            word_list[word] = pos_neg_ratio
    return word_list

#%%
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#%% Load Dataset
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#%% Split data in train and test
splitidx = 4000

train_pos  = all_positive_tweets[:splitidx]
train_neg  = all_negative_tweets[:splitidx]
train_x = train_pos + train_neg
train_x = [ process_tweet(tweet) for tweet in train_x ]
train_y = np.concatenate((np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1))), axis=0)

test_pos = all_positive_tweets[splitidx:]
test_neg = all_negative_tweets[splitidx:]
test_x = test_pos + test_neg
test_y = np.concatenate((np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1))), axis=0)

#%% Build frequencies
freqs, n_class = naive_bayes_build_freqs(train_x, train_y)

#%% Train
loglikelihood, logprior, logprobs = naive_bayes_train(train_x, train_y, (freqs, n_class))
acc = naive_bayes_accuracy(test_x, test_y, loglikelihood, logprior)

print(f"Accuracy: {acc}")


#%% Extract features
features = naive_bayes_extract_features(train_x, train_y, logprobs)
features_pos = features[features[:,2] == 1]
features_neg = features[features[:,2] == 0]

#%% Plot features
fig, ax = plt.subplots(figsize = (8, 8))
colors = ['red', 'green']
ax.scatter(features[:,0], features[:,1], c=[colors[int(k)] for k in features[:,2]], s = 0.1, marker='*')

confidence_ellipse(features_pos[:, 0], features_pos[:, 1], ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
confidence_ellipse(features_neg[:, 0], features_neg[:, 1], ax, n_std=2, edgecolor='orange')

# Print confidence ellipses of 3 std
confidence_ellipse(features_pos[:, 0], features_pos[:, 1], ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
confidence_ellipse(features_neg[:, 0], features_neg[:, 1], ax, n_std=3, edgecolor='orange', linestyle=':')
ax.set_xlim(-250,0)
ax.set_ylim(-250,0)
ax.set_xlabel("Positive") # x-axis label
ax.set_ylabel("Negative") # y-axis label
ax.legend()
plt.show()

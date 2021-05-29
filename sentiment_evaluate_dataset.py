#%% Load Modules
import random
import numpy as np
from sty import fg, bg, ef, rs
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
from sentiment_utils import *

#%% Dataset
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


#%% Visualize the dataset class distribution

labels = 'Positives', 'Negative'
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
fig, ax = plt.subplots(figsize = (8, 8))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Distribution of positive and negative sentiment")
fig.savefig("sentiment_chart.png", dpi=150, figsize=(8,8))
#fig.show()


#%% Print some random tweets
for x in range(10):
    if random.randint(0,1):
        print(fg.green + all_positive_tweets[random.randint(0,5000)])
    else:
        print(fg.red + all_negative_tweets[random.randint(0,5000)])
print(fg.rs)

# Sample tweet
tweet = all_positive_tweets[2277]
print(fg.green + tweet + fg.rs)
print(process_tweet(tweet))

#%% Merge data to analysis
all_tweets = np.append(all_positive_tweets, all_negative_tweets)
all_labels = np.append(np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets)))

#%% Calculate frequencies
freqs = build_freqs(all_tweets, all_labels)

#%% Keys to report
keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

report_data =  [ (word, freqs.get((word, 1), 0), freqs.get((word, 0), 0)) for word in keys ]

#%% Plot the report
# convert counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in report_data])
y = np.log([x[2] + 1 for x in report_data])

# Plot a dot for each pair of words
fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(x, y)
ax.set_xlabel("Log Positive count")
ax.set_ylabel("Log Negative count")
for i in range(0, len(report_data)):
    print(f'{report_data[i][0]}, {int(x[i])}, {int(y[i])}')
    ax.annotate(report_data[i][0], (x[i], y[i]), fontsize=12)
ax.plot([0, 9], [0, 9], color = 'red') # Plot the red line that divides the 2 areas.
ax.set_title('Key count per sentiment')
#fig.show()
fig.savefig("sentiment_distribution.png", dpi=150, figsize=(8,8))

#%% Evaluate features
#%% Extract the features for all inputs
m = len(all_tweets)
X = np.zeros((m, 3))
for i, tweet in enumerate(all_tweets):
    X[i, :] = extract_features(tweet, freqs)

#%% Find some nice tweets to show
pos_idx = X[:,1].argmax()
neg_idx = X[:,2].argmax()
meh_idx = np.where(np.all(X[:,1:] > 1000,axis=1) & (np.abs(X[:,1] - X[:,2]) < 100))
meh_idx = meh_idx[0][0]

max_sen = max(X[pos_idx,1], X[neg_idx,2])

pos_tweet = all_tweets[pos_idx]
pos_tags = process_tweet(pos_tweet)
pos_tweet = re.sub("(.{75})", "\\1\n", pos_tweet, 0, re.DOTALL)

neg_tweet = all_tweets[neg_idx]
neg_tags = process_tweet(neg_tweet)
neg_tweet = re.sub("(.{75})", "\\1\n", neg_tweet, 0, re.DOTALL)

meh_tweet = all_tweets[meh_idx]
meh_tags = process_tweet(meh_tweet)
meh_tweet = re.sub("(.{75})", "\\1\n", meh_tweet, 0, re.DOTALL)


#%% Visualize each sentence in feature space
fig, ax = plt.subplots(figsize = (8, 8))
fig.subplots_adjust(bottom=0.2)
ax.plot([0, max_sen], [0, max_sen], color = 'blue', zorder=-1)
ax.scatter(X[:,1], X[:,2],s=2, alpha=.2,color='black')
ax.scatter(*X[pos_idx,1:], color='green',label=' '.join(pos_tags))
ax.scatter(*X[neg_idx,1:], color='red',label=' '.join(neg_tags))
ax.scatter(*X[meh_idx,1:], color='magenta',label=' '.join(meh_tags))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Sentence positivity")
ax.set_ylabel("Sentence negativity")
#for i in range(0, len(report_data)):
#    print(f'{report_data[i][0]}, {int(x[i])}, {int(y[i])}')
#    ax.annotate(report_data[i][0], (x[i], y[i]), fontsize=12)
ax.set_title('Sentences in feature space')
fig.legend(bbox_to_anchor=(.5,0.01), loc="lower center",borderaxespad=0.2)
fig.savefig('sentiment_tweets.png', dpi=150, figsize=(8,8))
#fig.show()


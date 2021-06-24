#%%
import os
import random as rnd
import trax
import trax.layers as tl
import trax.fastmath as fm
import trax.fastmath.numpy as np
from trax.supervised import training

from utils import Layer, load_tweets, process_tweet

#%%
def tweet_to_tensor(tweet, vocab_dict, unk_token='__UNK__', verbose=False):
    return [ vocab_dict.get(w, vocab_dict[unk_token])  for w in process_tweet(tweet) ]

#%%
def build_vocab(sentences):
    vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}
    for tweet in sentences:
        processed_tweet = process_tweet(tweet)
        for word in processed_tweet:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab

#%%
def data_generator(data_positive, data_negative, batch_size, loop, vocab_dict, shuffle=False):
    assert batch_size % 2 == 0
    n_to_take = batch_size // 2

    positive_index = 0
    negative_index = 0
    len_data_positive = len(data_positive)
    len_data_negative = len(data_negative)

    positive_index_lines = list(range(len_data_positive))
    negative_index_lines = list(range(len_data_negative))

    if shuffle:
        rnd.shuffle(positive_index_lines)
        rnd.shuffle(negative_index_lines)

    stop = False
    while not stop:
        batch = []

        for i in range(n_to_take):
            if positive_index >= len_data_positive:
                if not loop:
                    stop = True
                    break
                positive_index = 0

            if shuffle:
                rnd.shuffle(positive_index_lines)

            tweet = data_positive[positive_index_lines[positive_index]]
            tensor = tweet_to_tensor(tweet, vocab_dict)
            batch.append(tensor)
            positive_index = positive_index + 1

        for i in range(n_to_take):
            if negative_index >= len_data_negative:
                if not loop:
                    stop = True
                    break
                negative_index = 0

            if shuffle:
                rnd.shuffle(negative_index_lines)

            tweet = data_negative[negative_index_lines[negative_index]]
            tensor = tweet_to_tensor(tweet, vocab_dict)
            batch.append(tensor)
            negative_index += 1

        if stop:
            break;

        positive_index += n_to_take
        negative_index += n_to_take

        # This padding just works for the actual batch, a future batch can have larger
        # elements... need to think about it.
        max_len = max([len(t) for t in batch])
        # max_len = 51

        tensor_pad_l = []
        for tensor in batch:
            n_pad = max_len - len(tensor)
            pad_l = [0] * n_pad
            tensor_pad = tensor + pad_l
            tensor_pad_l.append(tensor_pad)

        inputs = np.array(tensor_pad_l, dtype='int32')
        target_positive = [1] * (len(batch)//2)
        target_negative = [0] * (len(batch)//2)
        target_l = target_positive + target_negative
        targets = np.array(target_l, dtype='int32')
        example_weigths = np.ones_like(targets, dtype='int32')

        yield inputs, targets, example_weigths

#%% Create the training data generator
def train_generator(train_pos, train_neg, vocab, batch_size, shuffle=False):
    return data_generator(train_pos, train_neg, batch_size, True, vocab, shuffle)

#%% Create the validation data generator
def val_generator(val_pos, val_neg, vocab, batch_size, shuffle=False):
    return data_generator(val_pos, val_neg, batch_size, True, vocab, shuffle)

#%% Create the validation data generator
def test_generator(val_pos, val_net, vocab, batch_size, shuffle=False):
    return data_generator(val_pos, val_neg, batch_size, False, vocab, shuffle)


#%%
class Relu(Layer):
    def forward(self, x):
        activation = np.maximum(0, x)
        return activation

#%%
class Dense(Layer):
    def __init__(self, n_units, init_stdev=0.1):
        self._n_units = n_units
        self._init_stdev = init_stdev

    def forward(self, x):
        dense = x @ self.weights
        return dense

    def init_weights_and_state(self, input_signature, random_key):
        input_shape = input_signature.shape
        weights_shape = (input_shape[1], self._n_units)
        normal = tl.initializers.RandomNormalInitializer(stddev=self._init_stdev)
        w = normal(weights_shape, random_key)
        self.weights = w
        return self.weights

#%%
def classifier(vocab_size, embedding_dim=256, output_dim=2, mode='train'):
    """The learning model"""
    embed_layer = tl.Embedding(
            vocab_size=vocab_size,
            d_feature=embedding_dim)

    # TODO: 1 here? -1 ?
    mean_layer = tl.Mean(axis=0, keepdims=True)

    dense_outputlayer = tl.Dense(n_units = output_dim)

    log_softmax_layer = tl.LogSoftmax()

    model = tl.Serial(
            embed_layer,
            mean_layer,
            dense_output_layer,
            log_softmax_layer)

    return model

#%%
def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    training_loop = training.Loop(
        classifier,
        train_task,
        eval_tasks = eval_task,
        output_dir = output_dir)

    training_loop.run(n_steps = n_steps)

    return training_loop

#%%
if __name__ == '__main__':

    #%% Parameters
    batch_size = 16
    n_steps = 100
    split_pos = 4000
    output_dir = '~/model/'

    output_dir_expand = os.path.expanduser(output_dir)

    #%%
    # Load positive and negative tweets
    all_positive_tweets, all_negative_tweets = load_tweets()

    #%% Find max tweet
    # TODO: Tavez nao necessario

    #%%
    val_pos   = all_positive_tweets[split_pos:]
    train_pos  = all_positive_tweets[:split_pos]
    val_neg   = all_negative_tweets[split_pos:]
    train_neg  = all_negative_tweets[:split_pos]

    #%%
    train_x = train_pos + train_neg
    val_x  = val_pos + val_neg
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
    val_y  = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))

    #%%
    vocab = build_vocab(train_x)
    vocab_size = len(vocab)

    #%%
    # Testing your Dense layer
    fmrnd = fm.ops.RandomBackend()
    random_key = fmrnd.get_prng(seed=0)

    #%%
    rnd.seed(270)

    #%%
    train_task = training.TrainTask(
            labeled_data=train_generator(train_pos, train_neg, vocab, batch_size=batch_size, shuffle=True),
            loss_layer=tl.WeightedCategoryCrossEntropy(),
            optimizer=trax.optimizers.Adam(0.01),
            n_steps_per_checkpoint=10,
            )

    #%%
    eval_task = training.EvalTask(
            labeled_data=val_generator(val_pos, val_neg, vocab, batch_size=batch_size, shuffle=True),
            metrics=[tl.metrics.WeightedCategoryCrossEntropy(), tl.metrics.WeightedCategoryAccuracy()],
            )

    #%%
    model = classifier(vocab_size)

    #%%
    training_loop = train_model(model, train_task, eval_task, n_steps, output_dir_expand)

    #%%
#   tmp_embed = np.array([[1,2,3,],
#                       [4,5,6]
#                      ])
#
#   # take the mean along axis 0
#   print("The mean along axis 0 creates a vector whose length equals the vocabulary size")
#   display(np.mean(tmp_embed,axis=0))
#
#   print("The mean along axis 1 creates a vector whose length equals the number of elements in a word embedding")
#   display(np.mean(tmp_embed,axis=1))

#%% TEST1
# Get a batch from the train_generator and inspect.
# rnd.seed(30)
# inputs, targets, example_weights = next(train_generator(train_pos, train_neg, vocab, 4, shuffle=True))
#
# # this will print a list of 4 tensors padded with zeros
# print(f'Inputs: {inputs}')
# print(f'Targets: {targets}')
# print(f'Example Weights: {example_weights}')
#
#%% Test2
# Test the train_generator

# Create a data generator for training data,
# which produces batches of size 4 (for tensors and their respective targets)
# tmp_data_gen = train_generator(train_pos, train_neg, vocab, batch_size = 4)
#
# # Call the data generator to get one batch and its targets
#tmp_inputs, tmp_targets, tmp_example_weights = next(tmp_data_gen)
#
#print(f"The inputs shape is {tmp_inputs.shape}")
#print(f"The targets shape is {tmp_targets.shape}")
#print(f"The example weights shape is {tmp_example_weights.shape}")
#
#for i,t in enumerate(tmp_inputs):
#    print(f"input tensor: {t}; target {tmp_targets[i]}; example weights {tmp_example_weights[i]} shp {t.shape}")

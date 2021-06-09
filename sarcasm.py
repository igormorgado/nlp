## Read dataset
import json

with open("datasets/sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

## Some settings
oov_token = "<OOV>"
vocab_size = 10000
embedding_dim = 16
max_length = 100 
padding_type = 'post'
trunc_type = 'pre'
training_size_perc = .7
num_epochs = 30
verbosity = 2


# Split training
import numpy as np
training_size = int(len(sentences) * training_size_perc)

training_sentences = np.array(sentences[:training_size])
testing_sentences = np.array(sentences[training_size:])

training_labels = np.array(labels[:training_size])
testing_labels = np.array(labels[training_size:])

# Check if split is nice...
print(len(sentences) == len(training_sentences) + len(testing_sentences))
print(len(labels) == len(training_labels) + len(testing_labels))

# Set memory wild! ;-)
# for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Prepare the vocabulary just with training words
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts (training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences (testing_sentences)
testing_padded = pad_sequences(testing_sequences,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)


#  Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()



# Train
fit_hist = model.fit (training_padded, training_labels,
                      validation_data=(testing_padded, testing_labels),
                      epochs=num_epochs,
                      verbose=verbosity)


# Check for sentiment
sentence = [
        "granny starting to fears spiders in the garden might be real",
        "the weather today is bright and sunny"]

sequences = tokenizer.texts_to_sequences(sentence)

padded = pad_sequences (sequences, maxlen=max_length,
                        padding=padding_type, truncating=trunc_type)

print(model.predict(padded))

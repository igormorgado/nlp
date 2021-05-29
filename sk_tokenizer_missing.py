from sklearn.feature_extraction.text import CountVectorizer
from itertools import zip_longest


sentences = [
        'I love my dog',
        'I love my cat',
        'You love my dog',
        'Do you think my dog is amazing?'
        ]

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=100)
X = vectorizer.fit_transform(sentences)
tokenizer = vectorizer.build_tokenizer()
word_index = {v:k for k, v in enumerate(vectorizer.get_feature_names(), 2)}
word_index['<OOV>'] = 1
#sequences = [[ word_index[k] for k in tokenizer(s.lower()) if k in word_index ] for s in sentences ]
sequences = [[ word_index[k] if k in word_index else word_index['<OOV>'] for k in tokenizer(s.lower()) ] for s in sentences ]

padded_right = list(zip(*zip_longest(*sequences, fillvalue=0)))
## FALTA INVERTER CADA SEEQUENCIA
padded_left = list(zip(*itertools.zip_longest(*map(reversed, sequences), fillvalue=0)))

print (word_index)
print (sentences)
print (sequences)
print (padded_right)

test_data = [
        'i really love my dog',
        'my dog loves my manatee'
]

test_seq = [[ word_index[k] if k in word_index else word_index['<OOV>'] for k in tokenizer(s.lower()) ] for s in test_data ]
print(test_data)
print (test_seq)


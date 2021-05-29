from sklearn.feature_extraction.text import CountVectorizer

sentences = [
        'I love my dog',
        'I love my cat',
        'You love my dog',
        'Do you think my dog is amazing?'
        ]

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=100)
X = vectorizer.fit_transform(sentences)
tokenizer = vectorizer.build_tokenizer()
word_index = {v:k for k, v in enumerate(vectorizer.get_feature_names())}
sequences = [[ word_index[k.lower()] for k in tokenizer(s) if k in word_index ] for s in sentences ]

print(word_index)
print(sentences)
print(sequences)


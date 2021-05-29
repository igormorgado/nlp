from sklearn.feature_extraction.text import CountVectorizer

sentences = [
        'I love my dog',
        'I love my cat',
        ]

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(sentences)
word_index = {v:k for k, v in enumerate(vectorizer.get_feature_names())}

print(word_index)


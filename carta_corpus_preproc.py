## Read letter and join in a single string (as file contents)
with open('carta.txt', 'r') as fd:
    text_raw = []
    for line in fd.readlines():
        text_raw.append(line)
fd.close()
text_raw = ''.join(text_raw)

## Start pre-process. Remove stopwords and decapitalize
stopwords_ptbr = stopwords.words('portuguese')

text_pp = text_raw


## Creates the lists with text, paragraphs, sentences and words 
text = []
text_nosw = []
paragraphs = []
paragraphs_nosw = []
sentences = []
sentences_nosw = []
words = []
words_nosw = []

for paragraph in text_pp.split('\n\n'):
    paragraph = ' '.join(paragraph.split('\n')).strip()
    paragraph_tkn = word_tokenize(paragraph.lower())
    paragraph_nosw = [ w for w in paragraph_tkn if w not in stopwords_ptbr ]
    paragraph_nosw = ' '.join(paragraph_nosw)
    for sentence in paragraph.split('.'):
        sentence = sentence.strip()
        for word in sentence.split(' '):
            word = word.strip()
            if word != '':
                words.append(word)
        if sentence != '':
            sentences.append(sentence)
    if paragraph != '':
        paragraphs.append(paragraph)

    paragraphs_nosw.append(paragraph_nosw)
    sentences_nosw.extend(paragraph_nosw.split(' . '))
    words_nosw.extend(paragraph_nosw.split(' '))

text = text_pp.replace('\n', ' ').strip()
text_nosw = ' '.join(paragraphs_nosw)

text_nosw_nopunct = ''.join([ w for w in text_nosw if w not in [',', ':', '.', ' - ', ';', '–', ' — ', '—']])
text_nosw_nopunct = ' '.join(word_tokenize(text_nosw_nopunct))

##
freq = ngram_frequency(text_nosw_nopunct, 1)
total = compute_probability(freq)

##
print(freq)

##
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
bar1 = ax.bar(freq.keys(), freq.values())
#ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical', ha='right')


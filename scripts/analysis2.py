from collections import Counter
import io
from nltk import ngrams

f_0 = open('../data/train_0s.tsv', 'r')
documents_0 = f_0.read().splitlines()

f_1 = open('../data/train_1s.tsv', 'r')
documents_1 = f_1.read().splitlines()


# calculate words frequencies per document
word_frequencies_0 = [Counter(ngrams(document.lower().split(),2)) for document in documents_0]

word_frequencies_1 = [Counter(ngrams(document.lower().split(),2)) for document in documents_1]

# calculate document frequency
document_frequencies_0 = Counter()
map(document_frequencies_0.update, (word_frequency.keys() for word_frequency in word_frequencies_0))

document_frequencies_1 = Counter()
map(document_frequencies_1.update, (word_frequency.keys() for word_frequency in word_frequencies_1))

f_out = io.open('../analysis/bigrams_1.txt', 'wb')
f_keys = io.open('../analysis/bigrams_keys.txt', 'wb')
for k,v in document_frequencies_1.most_common(100000):
    # if v > document_frequencies_0[k] and document_frequencies_0[k] < 5 and v > 20 :
    ratio = document_frequencies_1[k]/float(0.1 + document_frequencies_0[k])
    if v > document_frequencies_0[k] and k[0] != '1' and document_frequencies_0[k] < 10 and ratio > 7 and v > 5:
        temp = '{0}, {1}, {2}\n'.format(k, v, document_frequencies_0[k])
        temp_key = '{0}\n'.format(k)
        f_keys.write(temp_key.decode('ascii', 'ignore'))
        f_out.write(temp.decode('ascii', 'ignore'))
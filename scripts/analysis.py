import nltk
import io

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f_0 = open('../data/train_0s.tsv', 'r')
raw_0 = f_0.read()

f_1 = open('../data/train_1s.tsv', 'r')
raw_1 = f_1.read()


tokens_0 = nltk.word_tokenize(raw_0.decode('utf-8'))
tokens_1 = nltk.word_tokenize(raw_1.decode('utf-8'))

fdist_0 = nltk.FreqDist(tokens_0)
fdist_1 = nltk.FreqDist(tokens_1)

f_out = io.open('../analysis/diff_0_1_select.txt', 'wb')
for k,v in fdist_1.most_common(100000):
    if fdist_1[k] > fdist_0[k] and fdist_0[k] < 50:
        temp = '{0}, {1}, {2}\n'.format(k, v, fdist_0[k])
        f_out.write(temp.decode('ascii', 'ignore'))

# f_out = io.open('../analysis/train_0_unigram.txt', 'wb')
# # compute frequency distribution for all the bigrams in the text
# fdist = nltk.FreqDist(tokens)
# for k,v in fdist.most_common(10000):
#     temp = '{0}, {1}\n'.format(k, v)
#     f_out.write(temp.decode('ascii', 'ignore'))
